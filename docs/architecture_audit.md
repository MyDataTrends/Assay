# Architecture Audit: Parallel Implementations in Assay
*Generated: 2026-03-09*

## Executive Summary

The Assay codebase contains a **dual-architecture pattern** where newer systems (Cascade Planner, LLM Manager, Smart Charts, Session Context) run in parallel with legacy systems (chatbot/intent_parser, orchestrate_workflow, scattered LLM calls, direct viz wrappers). This document inventories every such overlap and makes a binding decision on which to use going forward.

---

## 1. Intent Detection

### Implementations

**1A. Cascade Planner — USE THIS**
- File: `orchestration/cascade_planner.py` (lines 116–213)
- Regex patterns for 10 intents with confidence scores; LLM fallback if confidence < 0.5
- Intents: `DESCRIBE_DATA`, `VISUALIZE`, `TRANSFORM`, `FILTER`, `AGGREGATE`, `MODEL_TRAIN`, `MODEL_PREDICT`, `ENRICH_DATA`, `EXPORT`, `COMPARE`, `UNKNOWN`
- Exported as `orchestration.classify_intent()`
- Currently used in: `ui/chat_logic.py:77–111` (`cascade_detect_intent()`)

**1B. Chatbot Intent Parser — DEPRECATE**
- File: `chatbot/intent_parser.py` (lines 50–140)
- spaCy PhraseMatcher + keyword fallback + two-stage LLM classification
- Only 4 intents: `visualization`, `modeling`, `scenario`, `unknown`
- Used in: `chatbot/chatbot.py` (legacy chatbot interface only)

**1C. LLM Intent Classifier — DEPRECATE**
- File: `chatbot/llm_intent_classifier.py`
- `modeling_needed()` (binary LLM classification), `classify_modeling_type()` (multi-class)
- Called by `chatbot/intent_parser.py` — goes away when 1B is removed

### Decision
> **All intent detection routes through `orchestration.classify_intent()` (Cascade Planner).** The `chatbot/` module is legacy infrastructure. Add a `DeprecationWarning` to `chatbot/intent_parser.py` immediately; remove the module after all callsites are migrated.

| File | Line | Action |
|------|------|--------|
| `chatbot/intent_parser.py` | All | Add `DeprecationWarning`; migrate callers |
| `chatbot/llm_intent_classifier.py` | All | Delete when `intent_parser.py` is removed |
| `ui/chat_logic.py` | 77–226 | Already uses Cascade — good |
| `ui/dashboard.py` | Any `detect_intent` | Replace with `orchestration.classify_intent()` |

---

## 2. LLM Calling Interface

### Implementations

**2A. LLM Manager Interface — USE THIS**
- File: `llm_manager/llm_interface.py`
- `get_llm_completion()`, `get_llm_chat()`, `is_llm_available()`, `analyze_data_with_llm()`, `suggest_visualizations()`, `generate_insight()`
- Tries registry → subprocess fallback automatically
- Handles empty string returns gracefully

**2B. Direct Registry Access — INTERNAL ONLY**
- File: `llm_manager/registry.py`
- `registry.get_active_provider().complete(...)` pattern
- Acceptable in `cascade_planner.py` where raw provider access is needed; nowhere else

**2C. Subprocess Manager — INTERNAL ONLY**
- File: `llm_manager/subprocess_manager.py`
- Direct subprocess management of GGUF models
- Should never be imported directly outside `llm_interface.py`

**2D. Scattered Direct API Imports — REMOVE**
- `from anthropic import Anthropic` in agent code
- `from openai import OpenAI` anywhere
- These bypass fallback logic, error handling, and the registry

### Decision
> **All LLM calls go through `llm_manager.llm_interface`.** The registry and subprocess manager are implementation details of `llm_interface.py`. Direct `anthropic`/`openai` imports must be replaced.

| File | Action |
|------|--------|
| Any `from anthropic import` | Replace with `get_llm_completion()` or `get_llm_chat()` |
| Any `from openai import` | Replace with `get_llm_completion()` or `get_llm_chat()` |
| `llm_manager/subprocess_manager.py` | Keep, but mark module-private |

---

## 3. Chart / Visualization Generation

### Implementations

**3A. Smart Charts — USE FOR RECOMMENDATIONS**
- File: `visualization/smart_charts.py`
- `profile_column()` → column role detection (TEMPORAL, CATEGORICAL, NUMERIC_CONTINUOUS, etc.)
- `recommend_charts()` → 5 ranked chart types with confidence scores
- `get_best_chart()` → single best pick
- Deterministic rules engine; no LLM dependency

**3B. Tool Registry Chart Generator — USE FOR EXECUTION**
- File: `orchestration/tool_registry.py` (lines 230–257)
- `_handle_chart_generator()` → builds plotly figures from tool invocations
- This is the output layer; it consumes what 3A recommends

**3C. UI Visualizations Module — WRAPPER LAYER (OK)**
- File: `ui/visualizations.py`
- Plotly Express wrappers for Streamlit rendering
- Fine to keep as a thin rendering adapter; should not contain recommendation logic

**3D. MCP Visualization Tool — NEEDS REFACTOR**
- File: `mcp_server/tools/visualization.py` (lines 49–100)
- Has its own simple column-type detection and rule-based suggestions
- Duplicates Smart Charts logic at a lower quality level
- Should delegate recommendation logic to `smart_charts.recommend_charts()`

**3E. LLM-Based Visualization Suggestion — SUPPLEMENT ONLY**
- File: `llm_manager/llm_interface.py` (lines 212–292), `suggest_visualizations()`
- LLM generates suggestions from column context
- Should feed into Smart Charts as a fallback input, not replace the rules engine

**3F. `scripts/visualization_selector.py` — DEAD CODE, DELETE**
- File: `scripts/visualization_selector.py`
- `_heuristic_visualization()` — inline dtype-based rules (datetime→line, categorical+numeric→bar, etc.)
- `infer_visualization_type()` — heuristic with LLM fallback via `preprocessing.llm_preprocessor.llm_completion` (old LLM call pattern)
- **Problem**: Uses the old `llm_completion` from `preprocessing.llm_preprocessor` (not `llm_interface`), duplicates Smart Charts logic at much lower quality, no confidence structure
- **Callers**: None found — appears to be dead code from before Smart Charts existed

### Decision
> **Smart Charts is the recommendation engine. Tool Registry chart_generator is the execution engine. MCP tool must delegate to Smart Charts instead of re-implementing detection. LLM suggestions are a supplement when Smart Charts confidence is low. `scripts/visualization_selector.py` should be deleted — it is dead code that predates Smart Charts.**

```
User asks for chart
    → classify_intent() → VISUALIZE
    → smart_charts.recommend_charts(df)      ← recommendation
    → tool_registry._handle_chart_generator() ← execution
    → ui/visualizations.py (Streamlit render) ← display

MCP API path:
    → mcp_server/tools/visualization.py
    → delegate to smart_charts.recommend_charts()  ← NEEDS WIRING
```

---

## 4. Data Profiling

### Implementations

**4A. Tool Registry Profiler — PRIMARY for Cascade flows**
- File: `orchestration/tool_registry.py` (lines 100–130)
- Triggered by `DESCRIBE_DATA` intent; returns shape, nulls, dtypes, stats, memory

**4B. Metadata Parser — LEGACY (keep for old pipeline)**
- File: `preprocessing/metadata_parser.py`
- `pre_scan_metadata()`, `parse_metadata()`, `infer_column_meta()`
- Used in `orchestrate_workflow.py` (old pipeline)

**4C. Smart Charts Column Profiler — VISUALIZATION ONLY**
- File: `visualization/smart_charts.py` (lines 69–155)
- `profile_column()` / `profile_dataframe()`
- Scoped to visualization suitability — intentionally specialized, not a general profiler

**4D. LLM Preprocessor — INTELLIGENT SUMMARY**
- File: `preprocessing/llm_preprocessor.py`
- LLM-guided column understanding and cleaning suggestions

### Decision
> **These serve different purposes and should coexist, but the data model returned must be standardized.** All profilers should agree on column type vocabulary. Cascade flows use Tool Registry profiler. Old pipeline uses Metadata Parser. Smart Charts profiler stays visualization-scoped.

---

## 5. Session Management

### Implementations

**5A. Session Context — PRIMARY (in-memory UI state)**
- File: `ui/session_context.py`
- All Streamlit session state: chat history, datasets, LLM context, action log, discoveries
- Single source of truth during a session

**5B. Session DB — PERSISTENCE LAYER (supplement)**
- File: `storage/session_db.py`
- SQLite persistence for run history and crash recovery
- Should be read on startup to restore session context

**5C. Parquet Cache — FALLBACK**
- File: `storage/local_backend.py`
- Dataset backup for crash recovery

**5D. Session Persistence — SUPPLEMENTARY (structured snapshots)**
- File: `ui/session_persistence.py`
- `SessionPersistence.save_snapshot()` / `restore_snapshot()` — SQLite-backed snapshot with encryption option
- Structured summary schema: intent, tool sequence, outcome, deltas
- Weighted memory retrieval blending long-term summaries with short-term logs
- Distinct from `session_db.py` (run history) — this is user-triggered checkpoint/restore

### Decision
> **Session Context is primary in-memory state. Session Persistence adds user-controlled checkpoint/restore on top. Session DB is run-history logging. Parquet cache is dataset fallback. All four serve distinct purposes and should coexist. The gap is that `session_persistence.py` is likely not exposed in the dashboard UI — wire it to a "Save Session" button.**

---

## 6. Report Generation

### Implementations

**6A. Report Generator — PRIMARY**
- File: `reports/report_generator.py`
- Interactive HTML/PDF with CSS theming, metric cards, plotly.js embedding

**6B. Notebook Generator — SUPPLEMENTARY**
- File: `reports/notebook_generator.py`
- Jupyter notebook export

### Decision
> **Both can coexist — they serve different output formats.** No consolidation needed.

---

## 7. Main Workflow: Cascade vs Legacy

### Implementations

**7A. Cascade Planner — NEW PRIMARY**
- Files: `orchestration/cascade_planner.py`, `orchestration/tool_registry.py`
- Intent classify → dynamic plan → tool execution with retry + learning
- In-memory DataFrames; no file I/O
- Integrated learning loop (PlanLearner)

**7B. orchestrate_workflow — LEGACY**
- File: `orchestration/orchestrate_workflow.py`
- Static analyzer selection, file-based I/O (Parquet/CSV)
- Multi-stage pipeline: preprocess → enrich → analyze → output → agents
- No learning; no dynamic intent routing

**7C. run_workflow / WorkflowManager — BRIDGE LAYER**
- File: `orchestration/orchestrate_workflow.py` (lines 106–216)
- Strategy 1: LLM dynamic analysis
- Strategy 2: `orchestrate_workflow()` (old)
- Strategy 3: Minimal fallback

### Current wiring issues
| Location | Issue |
|----------|-------|
| `ui/dashboard.py:61` | Imports `orchestrate_workflow` directly |
| `ui/dashboard.py:206` | Calls `orchestrate_dashboard()` (legacy path) |
| `ui/chat_logic.py:186–226` | `should_use_cascade()` — confidence threshold of 0.8 gates Cascade; everything below falls to legacy |

### Decision
> **Cascade Planner is the target architecture. The confidence threshold in `should_use_cascade()` should be lowered (0.6 is reasonable) to route more traffic through Cascade. New features are built exclusively against Cascade. Legacy pipeline is kept for backward compatibility only and marked deprecated.**

---

## 8. Dashboard/UI Wiring: Intended Data Flow

The correct flow post-cleanup:

```
User message
    ↓
ui/chat_logic.py
    → cascade_detect_intent()          [orchestration/cascade_planner.py]
    → should_use_cascade() → True
    → cascade_execute()
        → get_planner().plan()         [orchestration/cascade_planner.py]
        → planner.execute()
            → tool_registry.*          [orchestration/tool_registry.py]
                → smart_charts.*       [visualization/smart_charts.py]  (if VISUALIZE)
                → data_profiler        (if DESCRIBE_DATA)
                → llm_interface.*      [llm_manager/llm_interface.py]   (if LLM needed)
    → Render result in Streamlit       [ui/visualizations.py]
```

Fallback (should_use_cascade() → False):
```
    → LLM code generation (temporary; migrate intents to Cascade to eliminate this path)
```

---

## Master Decision Table

| Area | Use This | Deprecate / Remove |
|------|----------|--------------------|
| Intent detection | `orchestration.classify_intent()` (Cascade) | `chatbot/intent_parser.py`, `chatbot/llm_intent_classifier.py` |
| LLM calling | `llm_manager.llm_interface` functions | Direct `anthropic`/`openai` imports anywhere |
| LLM registry | Internal to `llm_interface.py` | Direct `registry.get_active_provider()` outside orchestration |
| Chart recommendation | `visualization/smart_charts.recommend_charts()` | MCP viz tool's inline detection logic; `scripts/visualization_selector.py` (delete) |
| Chart execution | `orchestration/tool_registry._handle_chart_generator()` | — |
| Chart rendering (Streamlit) | `ui/visualizations.py` (keep as thin wrapper) | — |
| Data profiling (Cascade) | `orchestration/tool_registry._handle_data_profiler()` | — |
| Data profiling (legacy pipeline) | `preprocessing/metadata_parser.py` | When legacy pipeline is removed |
| Session state | `ui/session_context.py` | — |
| Session persistence | `storage/session_db.py` (supplement) | — |
| Main workflow | `orchestration/cascade_planner.py` | `orchestration/orchestrate_workflow.py` (long-term) |
| Reports | `reports/report_generator.py` | — |
| Notebooks | `reports/notebook_generator.py` | — |

---

## Prioritized Action Items

### Immediate (before next feature work)
1. **Intent detection**: Add `DeprecationWarning` in `chatbot/intent_parser.py`. Audit all callers and switch to `orchestration.classify_intent()`.
2. **Dashboard legacy call**: Fix `ui/dashboard.py:206` — replace `orchestrate_dashboard()` call with Cascade path.
3. **should_use_cascade threshold**: Lower from 0.8 → 0.6 in `ui/chat_logic.py:186` to expand Cascade coverage.
4. **Delete dead code**: Remove `scripts/visualization_selector.py` — zero callers found, predates Smart Charts, uses old LLM call pattern.

### Short-term
4. **MCP viz tool**: Refactor `mcp_server/tools/visualization.py` to call `smart_charts.recommend_charts()` instead of its inline column detection.
5. **Direct LLM imports**: Grep for `from anthropic import` and `from openai import` outside `llm_manager/`; replace with `llm_interface` calls.
6. **LLM viz fallback**: Wire `llm_interface.suggest_visualizations()` as a low-confidence fallback input into `smart_charts`, not a parallel path.

### Medium-term
7. **Cascade coverage expansion**: Map remaining intents that still fall through to code generation; add tool handlers for them.
8. **Legacy pipeline deprecation**: Add `DeprecationWarning` to `orchestrate_workflow()`. Write migration guide for any external callers.
9. **Profiler data model**: Standardize the dict schema returned by all profilers so downstream consumers can handle either source.

### Long-term
10. **Remove `chatbot/` module** after all migration is verified by tests.
11. **Remove `orchestrate_workflow()`** after Cascade covers all previously-handled intents.
12. **Integration test suite**: Tests that explicitly verify each user intent flows through the new architecture end-to-end.

---

## Key File Reference

| File | Role | Status |
|------|------|--------|
| `orchestration/cascade_planner.py` | Intent classification + plan execution | **Primary — maintain** |
| `orchestration/tool_registry.py` | Tool handlers for all operations | **Primary — maintain** |
| `orchestration/plan_learner.py` | Active learning on successful plans | **Primary — maintain** |
| `llm_manager/llm_interface.py` | Unified LLM calling interface | **Primary — maintain** |
| `visualization/smart_charts.py` | Chart recommendation engine | **Primary — maintain** |
| `ui/session_context.py` | In-memory session state | **Primary — maintain** |
| `ui/chat_logic.py` | User input routing | **Primary — maintain; lower threshold** |
| `ui/dashboard.py` | Streamlit entry point | **Fix legacy calls** |
| `chatbot/intent_parser.py` | Legacy intent parsing | **Deprecate** |
| `chatbot/llm_intent_classifier.py` | Legacy LLM intent wrapper | **Deprecate** |
| `orchestration/orchestrate_workflow.py` | Legacy workflow pipeline | **Deprecate long-term** |
| `mcp_server/tools/visualization.py` | MCP viz suggestions | **Refactor to use smart_charts** |
| `reports/report_generator.py` | HTML/PDF report output | **Keep** |
| `reports/notebook_generator.py` | Jupyter notebook export | **Keep** |
