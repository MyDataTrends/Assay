import logging
import os
import sys
import time
from pathlib import Path

# === Ensure project root is in Python path ===

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOG_DIR = Path(os.environ.get("LOG_DIR", str(_PROJECT_ROOT / "logs")))
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "app_debug.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Debug profiling
t0 = time.time()
PROFILE_LOG = _LOG_DIR / "startup_profile.log"

with open(PROFILE_LOG, "w", encoding="utf-8") as f:
    f.write(f"[{0.0:.3f}s] Startup: Begin imports\n")
    f.flush()

def log_profile(msg):
    try:
        with open(PROFILE_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.time()-t0:.3f}s] {msg}\n")
    except Exception as e:
        print(f"LOG ERROR: {e}")

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import faulthandler
faulthandler.enable()

import os
os.environ.setdefault("STREAMLIT_SERVER_ENABLECORS", "true")
log_profile("Importing streamlit...")
import streamlit as st

from reports.report_generator import generate_report_bytes

if "cache_cleared" not in st.session_state:
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state["cache_cleared"] = True
    except Exception:
        pass



log_profile("Importing pandas...")
import pandas as pd
import json
import hashlib
from feedback.ratings import store_rating
from feedback.role_corrections import store_role_corrections
log_profile("Importing ui modules...")
from ui.column_review import column_review
from orchestration.analysis_selector import select_analyzer
from storage.local_backend import load_datalake_dfs
from preprocessing.save_meta import (
    load_column_descriptions,
    load_column_roles,
)
log_profile("Importing metadata parser...")
from preprocessing.metadata_parser import infer_column_meta, merge_user_labels
log_profile("Importing llm_preprocessor...")
from preprocessing.llm_preprocessor import recommend_models_with_llm
from preprocessing.sanitize import scrub_df
log_profile("Importing visualizations...")
from ui.visualizations import (
    generate_bar_chart,
    generate_scatter_plot,
    generate_histogram,
    generate_heatmap,
    generate_pie_chart,
    generate_area_chart,
)
from orchestration.data_quality_scorer import summarize_for_display
from ui import redaction_banner
from ui.exploratory_tab import render_exploratory_tab
from ui.action_center import render_action_center
from ui.llm_settings import render_llm_settings, render_llm_settings_compact
from reports.report_generator import generate_report_bytes

# Unified session context
from ui.session_context import get_context, migrate_legacy_state

# New UX components
from ui.components.execution_timeline import render_execution_timeline
from ui.components.result_card import render_result_card
from ui.components.code_inspector import render_code_inspector
from ui.components.session_panel import render_session_panel

log_profile("Imports complete!")

migrate_legacy_state()


# ─────────────────────────────────────────────
# Helper functions (unchanged)
# ─────────────────────────────────────────────

def _hash_df(df: pd.DataFrame) -> str:
    data = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def _persist_dataset(name: str, df: pd.DataFrame):
    """Save dataset to Parquet cache for crash recovery."""
    try:
        cache_dir = Path("mcp_data/session_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_dir / f"{name}.parquet", index=False)
    except Exception:
        pass  # Non-critical — best-effort persistence


def _available_identifiers(df: pd.DataFrame, dest_dir: str = "metadata") -> list:
    h = _hash_df(df)
    path = Path(dest_dir)
    ids = []
    pattern = f"*_{h}_roles.json"
    for p in path.glob(pattern):
        stem = p.stem
        if stem.count("_") >= 2:
            ids.append(stem.split("_")[0])
    return sorted(set(ids))


def _analysis_suggestions(df: pd.DataFrame, meta) -> list:
    try:
        rec = ""
    except Exception:
        rec = ""
    suggestions: list = []
    if rec and "LLM unavailable" not in rec:
        suggestions = [s.strip("- ") for s in rec.splitlines() if s.strip()]
    if not suggestions:
        roles = {m.role for m in meta}
        if any(r in {"date", "time"} for r in roles):
            suggestions.append("Looks like a time-series – forecast sales?")
        if "categorical" in roles:
            suggestions.append("Try a classification model?")
        if "numeric" in roles:
            suggestions.append("Maybe run regression analysis?")
    return suggestions


mock_data = pd.DataFrame(
    {
        "Date": pd.date_range(start="2022-01-01", periods=100, freq="ME"),
        "Sales": [i * 1.05 for i in range(100)],
    }
)


def generate_line_chart(data):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(data["Date"], data["Sales"], label="Sales Over Time")
    plt.title("Line Chart of Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    st.pyplot(plt)


visualizations = {
    "line_chart": generate_line_chart,
    "bar_chart": generate_bar_chart,
    "scatter_plot": generate_scatter_plot,
    "histogram": generate_histogram,
    "heatmap": generate_heatmap,
    "pie_chart": generate_pie_chart,
    "area_chart": generate_area_chart,
}

def build_dashboard(data: pd.DataFrame, result: dict, target_column: str) -> dict:
    return {}

def generate_scenario(params):
    return f"Simulated scenario with adjustments: {params}"


# ─────────────────────────────────────────────
# Page config & title
# ─────────────────────────────────────────────

st.title("Assay")
st.caption("Local-first autonomous data analyst")


# ─────────────────────────────────────────────
# Run history (unchanged loading logic)
# ─────────────────────────────────────────────

history_file = Path(os.getenv("LOCAL_DATA_DIR", "local_data")) / "run_history.json"
run_history = []
if os.getenv("DYNAMO_SESSIONS_TABLE"):
    from storage import session_db
    for sess in session_db.list_sessions():
        meta = session_db.get_run_by_id(sess["run_id"])
        if meta:
            run_history.append(meta)
else:
    try:
        run_history = json.loads(history_file.read_text())
    except Exception:
        run_history = []

def load_ui_settings():
    settings_file = Path(os.getenv("LOCAL_DATA_DIR", "local_data")) / "settings.json"
    if settings_file.exists():
        try:
            return json.loads(settings_file.read_text())
        except Exception:
            return {}
    return {}

def save_ui_settings(settings):
    settings_file = Path(os.getenv("LOCAL_DATA_DIR", "local_data")) / "settings.json"
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        settings_file.write_text(json.dumps(settings))
    except Exception:
        pass

if "ui_settings" not in st.session_state:
    st.session_state["ui_settings"] = load_ui_settings()
    st.session_state["onboarding_dismissed"] = st.session_state["ui_settings"].get("onboarding_dismissed", False)
    st.session_state["seen_learning_tutorial"] = st.session_state["ui_settings"].get("seen_learning_tutorial", False)


# ─────────────────────────────────────────────
# Secure key storage (auto-load into env)
# ─────────────────────────────────────────────

try:
    from utils.key_storage import KeyStorage
    key_store = KeyStorage()
    for svc in key_store.list_services():
        key = key_store.get_key(svc)
        if key:
            if svc == "fred":
                os.environ["FRED_API_KEY"] = key
            if svc == "alphavantage":
                os.environ["ALPHAVANTAGE_API_KEY"] = key
            if svc == "openai":
                os.environ["OPENAI_API_KEY"] = key
except Exception:
    key_store = None

# ─────────────────────────────────────────────
# Dataset initialization
# ─────────────────────────────────────────────

if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}
if "primary_dataset_id" not in st.session_state:
    st.session_state["primary_dataset_id"] = None

# Auto-restore datasets from Parquet cache (crash recovery)
if not st.session_state["datasets"]:
    _cache_dir = Path("mcp_data/session_cache")
    if _cache_dir.exists():
        for _pq in _cache_dir.glob("*.parquet"):
            try:
                st.session_state["datasets"][_pq.stem] = pd.read_parquet(_pq)
            except Exception:
                pass
        if st.session_state["datasets"]:
            st.session_state["primary_dataset_id"] = list(st.session_state["datasets"].keys())[0]
            st.toast("♻️ Restored datasets from previous session")

# First-run: auto-load sample data so the dashboard isn't empty
if not st.session_state["datasets"]:
    _sample = Path("datasets/sample_sales_data.csv")
    if _sample.exists():
        try:
            from preprocessing.data_cleaning import standardize_dataframe
            _df = standardize_dataframe(pd.read_csv(_sample))
            st.session_state["datasets"]["Sample Sales"] = _df
            st.session_state["primary_dataset_id"] = "Sample Sales"
            _persist_dataset("Sample Sales", _df)
            st.toast("📊 Loaded sample sales data — explore or upload your own!")
        except Exception:
            pass

if st.session_state["datasets"]:
    active_ds = st.session_state["primary_dataset_id"] or list(st.session_state["datasets"].keys())[0]
    if active_ds not in st.session_state["datasets"]:
        active_ds = list(st.session_state["datasets"].keys())[0]
    st.session_state["primary_dataset_id"] = active_ds
    data = st.session_state["datasets"][active_ds]
else:
    data = None

# ─────────────────────────────────────────────
# Sidebar — Session Panel
# ─────────────────────────────────────────────

# LLM status
try:
    from llm_manager.registry import get_registry
    _llm_registry = get_registry()
    _active_model = _llm_registry.get_active_model()
    if _active_model:
        st.sidebar.success(f"AI: {_active_model.name}")
    else:
        st.sidebar.warning("No LLM selected")
except Exception:
    st.sidebar.info("LLM: Setup needed")

st.sidebar.divider()

# Session panel (datasets, saved workflows, recent queries, advanced toggle)
ctx = get_context()
with st.sidebar:
    render_session_panel(ctx)

    # Guided onboarding — show on first run until dismissed
    if not st.session_state.get("onboarding_dismissed"):
        with st.expander("🚀 **Getting Started**", expanded=True):
            st.markdown(
                "1. **Explore** the pre-loaded sample data below\n"
                "2. **Ask a question** in the chat — try *\"show trends\"*\n"
                "3. **Upload your own CSV** via the Data tab\n"
                "4. **Visualize** with the Explore tab"
            )
            if st.button("✓ Got it!", key="dismiss_onboarding"):
                st.session_state["onboarding_dismissed"] = True
                st.session_state["ui_settings"]["onboarding_dismissed"] = True
                save_ui_settings(st.session_state["ui_settings"])
                st.rerun()

# ─────────────────────────────────────────────
# Metadata loading (only when data is available)
# ─────────────────────────────────────────────

meta = {}
descriptions = {}
preview = None
if data is not None:
    ids = _available_identifiers(data)
    sel = None
    if ids:
        choice = st.sidebar.selectbox("Metadata version", ["Most recent"] + ids)
        if choice != "Most recent":
            sel = choice

    roles = load_column_roles(data, identifier=sel) or {}
    descriptions = load_column_descriptions(data, identifier=sel) or {}
    meta = infer_column_meta(data, descriptions)
    if roles:
        meta = merge_user_labels(meta, roles)
    st.session_state["column_meta"] = meta
    st.session_state["column_descriptions"] = descriptions

    preview = scrub_df(data.head(50))
    if descriptions:
        preview["Description"] = preview.index.map(lambda c: descriptions.get(c, ""))

    # Column review banner
    if st.session_state.get("needs_role_review") and not st.session_state.get("show_column_review"):
        st.warning("Accuracy is low. Clarify column meanings to improve?")
        if st.button("Review Columns"):
            st.session_state["show_column_review"] = True

    if st.session_state.get("show_column_review"):
        new_roles = column_review(data, st.session_state.get("column_meta", []))
        if new_roles:
            st.session_state["user_roles"] = new_roles
            store_role_corrections(data, new_roles)

# ─────────────────────────────────────────────
# Main tab structure — 3 tabs
# ─────────────────────────────────────────────

analyze_tab, data_tab, explore_tab, settings_tab = st.tabs(["Analyze", "Data", "Explore", "Settings"])

# ═══════════════════════════════════════
# ANALYZE TAB — Chat + Output + Timeline
# ═══════════════════════════════════════

with analyze_tab:
    col_main, col_right = st.columns([3, 1])

    # ── Right panel: execution timeline ──────────────────────────────────
    with col_right:
        trace = st.session_state.get("execution_trace", [])
        intent_last = st.session_state.get("last_intent")
        plan_type_last = st.session_state.get("last_plan_type")
        confidence_last = st.session_state.get("last_confidence")
        advanced = st.session_state.get("advanced_mode", False)
        render_execution_timeline(
            trace,
            advanced=advanced,
            intent=intent_last,
            plan_type=plan_type_last,
            confidence=confidence_last,
        )

    # ── Center panel: chat + outputs ─────────────────────────────────────
    with col_main:
        if data is None:
            st.markdown("### Welcome to Assay")
            st.markdown("**Your autonomous, local-first data analyst is ready.**")
            
            st.info("""
**What Assay can do for you:**
- 📈 **Detect Trends & Anomalies:** Instantly find hidden patterns in your data.
- 📊 **Smart Visualizations:** Ask for a chart in plain English, and Assay builds it.
- 🔮 **Forecasting & Modeling:** Run predictive models without writing a line of code.
- 🔒 **100% Private:** Your data never leaves your machine.
            """)
            st.write("Load a dataset below or upload your own in the **Data** tab to get started.")

            c1, c2 = st.columns([1, 2])
            with c1:
                if st.button("Load Sample Data", type="primary", use_container_width=True):
                    try:
                        from preprocessing.data_cleaning import standardize_dataframe
                        sample_path = Path("datasets/sample_sales_data.csv")
                        if sample_path.exists():
                            _df = pd.read_csv(sample_path)
                            _df = standardize_dataframe(_df)
                            st.session_state["datasets"]["Sample Sales"] = _df
                            st.session_state["primary_dataset_id"] = "Sample Sales"
                            _persist_dataset("Sample Sales", _df)
                            st.rerun()
                        else:
                            st.error(f"Sample data not found at {sample_path}")
                    except Exception as e:
                        st.error(f"Failed to load sample: {e}")
            with c2:
                st.info("Or go to the **Data** tab to upload your own CSV.")

        # Suggested analyses
        if "analysis_suggestions" not in st.session_state:
            st.session_state["analysis_suggestions"] = _analysis_suggestions(data, meta)

        suggestions = st.session_state["analysis_suggestions"]
        if suggestions:
            with st.expander("Suggested analyses", expanded=False):
                cols = st.columns(min(len(suggestions), 3))
                for i, sugg in enumerate(suggestions):
                    with cols[i % 3]:
                        if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
                            st.session_state["suggestion"] = sugg
                            st.rerun()

        # Import chat functions
        try:
            from ui.chat_logic import (
                detect_intent,
                should_use_cascade,
                cascade_execute,
                count_tokens,
                generate_visualization_code,
                generate_analysis_code,
                generate_informational_response,
                safe_execute,
                safe_execute_viz,
                generate_natural_answer,
                fallback_visualization,
                is_llm_ready,
                execute_analysis_with_retry,
            )
            from llm_learning.interaction_logger import get_interaction_logger, InteractionType
            interaction_logger = get_interaction_logger()
            chat_available = True
        except ImportError as e:
            chat_available = False
            interaction_logger = None
            st.warning(f"Chat functionality not available: {e}")

        # First-run tutorial
        if "seen_learning_tutorial" not in st.session_state:
            st.session_state.seen_learning_tutorial = False
        if not st.session_state.seen_learning_tutorial:
            with st.expander("Your AI gets smarter over time", expanded=True):
                st.markdown(
                    "Every question and rating improves Assay's responses for your data patterns. "
                    "Rate answers in the sidebar after each session."
                )
                if st.button("Got it!", key="dismiss_tutorial"):
                    st.session_state.seen_learning_tutorial = True
                    st.session_state["ui_settings"]["seen_learning_tutorial"] = True
                    save_ui_settings(st.session_state["ui_settings"])
                    st.rerun()

        if chat_available:
            llm_ready = is_llm_ready()

            if not llm_ready:
                st.warning("LLM unavailable — configure one in the **Settings** tab")

            # Build chat context
            primary_id = st.session_state.get("primary_dataset_id", "")
            context_parts = []
            datasets_all = st.session_state.get("datasets", {})
            if datasets_all:
                context_parts.append(f"Primary Dataset: {primary_id}")
                context_parts.append("Available Datasets and Schemas:")
                for ds_name, df in datasets_all.items():
                    is_primary = "(Primary)" if ds_name == primary_id else ""
                    schema_str = (
                        f"Dataset: {ds_name} {is_primary}\n"
                        f"Columns: {', '.join(df.columns)}\n"
                        f"Types: {df.dtypes.to_dict()}\n"
                        f"Sample:\n{df.head(3).to_string()}\n---"
                    )
                    context_parts.append(schema_str)
            ai_summary = st.session_state.get("ai_summary")
            if ai_summary:
                context_parts.append(f"AI Summary: {ai_summary}")
            chart_data = st.session_state.get("chart_suggestions", {})
            if "suggestions" in chart_data and chart_data["suggestions"]:
                sugs = [s.get("title", "") for s in chart_data["suggestions"][:3]]
                context_parts.append(f"Suggested visualizations: {', '.join(sugs)}")
            if ctx.chat_history:
                history_str = "\nRECENT CONVERSATION:\n"
                for msg in ctx.chat_history[-5:]:
                    history_str += f"{msg['role'].upper()}: {msg['content']}\n"
                context_parts.append(history_str)
            chat_context = ". ".join(context_parts)

            # Chat history display
            for i, msg in enumerate(ctx.chat_history):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    metadata = msg.get("metadata", {})
                    if "data" in metadata and metadata["data"] is not None:
                        msg_data = metadata["data"]
                        if hasattr(msg_data, 'show'):
                            try:
                                st.plotly_chart(msg_data, width="stretch", key=f"chat_fig_{i}")
                            except Exception:
                                st.error("Chart could not be rendered due to invalid data format.")
                        elif isinstance(msg_data, pd.DataFrame):
                            st.dataframe(msg_data)
                        elif isinstance(msg_data, (dict, list)):
                            with st.expander("View raw data", expanded=False):
                                st.json(msg_data)
                        elif isinstance(msg_data, (int, float, str, bool)):
                            if str(msg_data).strip():
                                st.write(msg_data)

            # Handle pending query from session panel "re-run" buttons
            pending_q = st.session_state.pop("pending_query", None)

            # Chat input
            if prompt := (pending_q or st.chat_input("Ask about your data...")):
                ctx.add_message("user", prompt)

                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    if not llm_ready:
                        response = "LLM not configured. Go to the Settings tab to set up a model."
                        st.markdown(response)
                        ctx.add_message("assistant", response)
                    else:
                        with st.status("Processing request...", expanded=True) as status:
                            # Give a rough token estimate to the user
                            estimated_tokens = count_tokens(prompt + chat_context)
                            if estimated_tokens > 2000:
                                st.toast(f"This is a complex request (~{estimated_tokens} tokens). It might take a moment.", icon="⏳")
                                
                            status.write("Identifying intent...")
                            
                            # NEW: 2026 Execution Hardening path
                            if should_use_cascade(prompt):
                                status.write("Routing to Cascade Planner...")
                                cascade_result = cascade_execute(data, prompt, context={"chat_history": ctx.chat_history})
                                
                                st.session_state["last_intent"] = cascade_result.get("intent", "cascade")
                                st.session_state["execution_trace"] = cascade_result.get("steps", [])
                                
                                if cascade_result["success"]:
                                    status.update(label="Cascade execution complete", state="complete", expanded=False)
                                    output = cascade_result.get("output", "Task completed.")
                                    # Fallback if no output
                                    if not output: output = "I've processed your request using the Cascade Planner."
                                    
                                    if isinstance(output, str):
                                        st.markdown(output)
                                        ctx.add_message("assistant", output)
                                    else:
                                        natural = generate_natural_answer(prompt, output)
                                        response = natural or "I've processed your request. Expand the data view below to see the detailed results."
                                        
                                        st.markdown(response)
                                        
                                        if isinstance(output, pd.DataFrame):
                                            st.dataframe(output)
                                        elif isinstance(output, (dict, list)):
                                            with st.expander("View results data", expanded=False):
                                                st.json(output)
                                        else:
                                            st.write(output)
                                            
                                        ctx.add_message("assistant", response, metadata={"data": output})
                                else:
                                    status.update(label="Cascade execution failed", state="error", expanded=True)
                                    st.warning("Cascade Planner encountered an error.")
                                    error_msg = cascade_result.get("error", "Unknown error")
                                    st.caption(f"Reason: {error_msg}")
                                    ctx.add_message("assistant", f"Cascade planning failed: {error_msg}")
                            
                            else:
                                intent = detect_intent(prompt, context=chat_context)
    
                                # Store intent for timeline
                                st.session_state["last_intent"] = intent
    
                                if intent == "visualization":
                                    status.write("Generating visualization code...")
                                    code = generate_visualization_code(
                                        data, prompt,
                                        context=chat_context,
                                        datasets=datasets_all,
                                    )
                                    fig = None
                                    if code:
                                        status.write("Rendering chart...")
                                        success, fig, error = safe_execute_viz(
                                            code, data, datasets=datasets_all
                                        )
                                    if not fig:
                                        status.write("Trying fallback renderer...")
                                        fig = fallback_visualization(data, prompt)

                                    if fig:
                                        status.update(label="Visualization ready", state="complete", expanded=False)
                                        # Result card
                                        render_result_card(
                                            title="Visualization",
                                            dataset=primary_id,
                                            code=code,
                                        )
                                        try:
                                            # Support both single figures and arrays of figures (e.g. from complex LLM generation)
                                            if isinstance(fig, list):
                                                for idx, f in enumerate(fig):
                                                    st.plotly_chart(f, width="stretch", key=f"viz_{hash(prompt)}_{idx}")
                                            else:
                                                st.plotly_chart(fig, width="stretch", key=f"viz_{hash(prompt)}")
                                        except Exception:
                                            st.error("Chart could not be rendered due to invalid data format.")
                                        
                                        ctx.add_message("assistant", "Here's your visualization:", metadata={"data": getattr(fig, 'to_dict', lambda: fig)() if not isinstance(fig, list) else None, "code": code})
                                        if interaction_logger:
                                            interaction_logger.log(
                                                prompt=prompt,
                                                response="Visualization created",
                                                interaction_type=InteractionType.VISUALIZATION,
                                                code_generated=code or "",
                                                execution_success=True,
                                                dataset_name=primary_id,
                                            )
                                        # Minimal trace for timeline
                                        st.session_state["execution_trace"] = [
                                            {"step_num": 1, "label": "Classify intent", "tool": "intent_detector", "status": "success", "duration_ms": 0},
                                            {"step_num": 2, "label": "Generate viz code", "tool": "llm", "status": "success", "duration_ms": 0},
                                            {"step_num": 3, "label": "Render chart", "tool": "plotly", "status": "success", "duration_ms": 0},
                                        ]
                                    else:
                                        status.update(label="Visualization failed", state="error", expanded=True)
                                        response = "I couldn't create that visualization. Try describing what you want to see more specifically."
                                        st.warning(f"Limited confidence in result — visualization could not be rendered.")
                                        st.info("Suggested next step: Try 'show a bar chart of [column] by [column]'.")
                                        ctx.add_message("assistant", response)
                                        if interaction_logger:
                                            interaction_logger.log(
                                                prompt=prompt,
                                                response=response,
                                                interaction_type=InteractionType.VISUALIZATION,
                                                code_generated=code or "",
                                                execution_success=False,
                                                dataset_name=primary_id,
                                            )

                                elif intent == "informational":
                                    status.write("Consulting knowledge base...")
                                    response = generate_informational_response(prompt, context=chat_context)
                                    status.update(label="Response ready", state="complete", expanded=False)
                                    if not response:
                                        response = "Unable to generate a response right now. Please check if the LLM is configured."
                                    st.markdown(response)
                                    ctx.add_message("assistant", response)
                                    st.session_state["execution_trace"] = [
                                        {"step_num": 1, "label": "Classify intent", "tool": "intent_detector", "status": "success", "duration_ms": 0},
                                        {"step_num": 2, "label": "Generate response", "tool": "llm", "status": "success", "duration_ms": 0},
                                    ]
                                    if interaction_logger:
                                        interaction_logger.log(
                                            prompt=prompt,
                                            response=response,
                                            interaction_type=InteractionType.CHAT,
                                            dataset_name=primary_id,
                                        )

                                else:
                                    status.write("Generating analysis code...")
                                    success, result_val, code, error = execute_analysis_with_retry(
                                        data, prompt, context=chat_context, datasets=datasets_all
                                    )

                                    if success:
                                        status.update(label="Analysis complete", state="complete", expanded=False)
                                        natural = generate_natural_answer(prompt, result_val)
                                        response = natural or "Here's what I found:"

                                        # Result card wraps the output
                                        render_result_card(
                                            title=prompt[:80],
                                            dataset=primary_id,
                                            insight=natural[:120] if natural else None,
                                            code=code,
                                        )
                                        st.markdown(response)
                                        if result_val is not None:
                                            if isinstance(result_val, pd.DataFrame):
                                                st.dataframe(result_val)
                                            elif isinstance(result_val, (dict, list)):
                                                with st.expander("View raw data", expanded=False):
                                                    st.json(result_val)
                                            elif isinstance(result_val, (int, float, str, bool)):
                                                if str(result_val).strip():
                                                    st.write(result_val)
                                        ctx.add_message("assistant", response, metadata={"data": result_val, "code": code})
                                        st.session_state["execution_trace"] = [
                                            {"step_num": 1, "label": "Classify intent", "tool": "intent_detector", "status": "success", "duration_ms": 0},
                                            {"step_num": 2, "label": "Generate analysis code", "tool": "llm", "status": "success", "duration_ms": 0},
                                            {"step_num": 3, "label": "Execute analysis", "tool": "python_exec", "status": "success", "duration_ms": 0},
                                        ]
                                        if interaction_logger:
                                            interaction_logger.log(
                                                prompt=prompt,
                                                response=response,
                                                interaction_type=InteractionType.ANALYSIS,
                                                code_generated=code,
                                                execution_success=True,
                                                dataset_name=primary_id,
                                            )
                                    else:
                                        status.update(label="Analysis failed", state="error", expanded=True)
                                        st.warning("Limited confidence in result — analysis could not complete.")
                                        st.caption(f"Reason: {error}")
                                        st.info("Suggested next step: Try rephrasing your question or check that the relevant columns exist in your dataset.")
                                        response = f"Analysis failed: {error}"
                                        ctx.add_message("assistant", response)
                                        st.session_state["execution_trace"] = [
                                            {"step_num": 1, "label": "Classify intent", "tool": "intent_detector", "status": "success", "duration_ms": 0},
                                            {"step_num": 2, "label": "Generate analysis code", "tool": "llm", "status": "success", "duration_ms": 0},
                                            {"step_num": 3, "label": "Execute analysis", "tool": "python_exec", "status": "failed", "duration_ms": 0, "error": error},
                                        ]
                                        if interaction_logger:
                                            interaction_logger.log(
                                                prompt=prompt,
                                                response=response,
                                                interaction_type=InteractionType.ANALYSIS,
                                                code_generated=code,
                                                execution_success=False,
                                                dataset_name=primary_id,
                                            )
                        st.rerun()

        # Export report
        st.divider()
        with st.expander("Export analysis report", expanded=False):
            from datetime import datetime
            col1, col2 = st.columns([3, 1])
            with col1:
                default_title = f"Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                report_title = st.text_input("Report title", value=default_title)
            with col2:
                if st.button("Prepare"):
                    figs = []
                    for msg in ctx.chat_history:
                        if "metadata" in msg and "data" in msg["metadata"] and msg["metadata"]["data"] is not None:
                            if hasattr(msg["metadata"]["data"], "to_json"):
                                figs.append(msg["metadata"]["data"])
                    try:
                        from reports.notebook_generator import generate_notebook_bytes
                        report_bytes = generate_report_bytes(
                            df=data, result=None, figures=figs,
                            chat_history=ctx.chat_history,
                            title=report_title,
                        )
                        st.session_state["ready_report"] = report_bytes
                        st.session_state["ready_notebook"] = generate_notebook_bytes(
                            chat_history=ctx.chat_history,
                            title=report_title,
                        )
                    except Exception as e:
                        st.error(f"Report generation failed: {e}")
            if "ready_report" in st.session_state:
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="Download HTML report",
                        data=st.session_state["ready_report"],
                        file_name=f"assay_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                        mime="text/html",
                        use_container_width=True,
                    )
                with dl_col2:
                    if "ready_notebook" in st.session_state:
                        st.download_button(
                            label="Download Jupyter Notebook",
                            data=st.session_state["ready_notebook"],
                            file_name=f"assay_notebook_{datetime.now().strftime('%Y%m%d_%H%M')}.ipynb",
                            mime="application/x-ipynb+json",
                            use_container_width=True,
                        )

        # Rating
        st.divider()
        with st.expander("Rate this session", expanded=False):
            rating = st.slider("Score", 1, 5, 3, key="rating_slider")
            if st.button("Submit", key="submit_rating"):
                store_rating(rating)
                st.success("Thanks for your feedback!")

        # Analysis results from non-chat workflow runs
        result = st.session_state.get("result")
        if result:
            target_col = result.get("model_info", {}).get("target") or (
                data.columns[-1] if data is not None and len(data.columns) else ""
            )

            diagnostics = result.get("diagnostics", {})
            if diagnostics:
                quality_summary = summarize_for_display(diagnostics)
                with st.expander(
                    f"{quality_summary['status_emoji']} Data Quality Report",
                    expanded=quality_summary['status'] != 'good',
                ):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Quality Score", f"{quality_summary['score']}%")
                    with col2:
                        st.metric("Data Completeness", f"{quality_summary['completeness_pct']}%")
                    with col3:
                        st.metric("Complete Rows", f"{quality_summary['rows_complete_pct']}%")
                    if quality_summary['warnings']:
                        st.warning("**Warnings:**")
                        for warning in quality_summary['warnings']:
                            st.write(f"- {warning}")
                    if quality_summary['high_risk_columns']:
                        st.error(f"**High-risk columns:** {', '.join(quality_summary['high_risk_columns'])}")
                    if quality_summary['missing_columns']:
                        with st.expander("Columns with missing data"):
                            st.write(", ".join(quality_summary['missing_columns']))
                    if quality_summary['proceed_with_caution']:
                        st.info("Consider reviewing column roles or providing additional data.")

            dash = build_dashboard(data, result, target_col)
            if dash.get("chart_result") is not None:
                st.write(dash["chart_result"])

            st.subheader("Analysis Results")
            decision = result.get("modeling_decision")
            if decision and not decision.get("modeling_required", True):
                st.info(f"No predictive model was run. Reason: {decision.get('reasoning', '')}")
            elif decision and decision.get("modeling_required"):
                st.info(f"Predictive modeling applied: {decision.get('modeling_type', '')}. Reason: {decision.get('reasoning', '')}")
            if result.get("modeling_failed"):
                st.warning("Limited confidence in result — modeling failed, showing descriptive results only.")
                if result.get("failure_reason"):
                    st.caption(f"Reason: {result['failure_reason']}")
                    st.info("Suggested next step: Try a different analysis type or check your dataset for sufficient rows.")

            atype = result.get("analysis_type")
            if atype == "regression":
                st.line_chart(result.get("predictions"))
            elif atype == "classification":
                st.json(result.get("report"))
            elif atype == "forecasting":
                st.line_chart(result.get("forecast"))
            elif atype == "clustering":
                st.write("Cluster Centers")
                st.write(result.get("centers"))
                if data is not None and data.shape[1] >= 2:
                    df_plot = data.iloc[:, :2].copy(deep=False)
                    df_plot["label"] = result.get("labels")
                    st.scatter_chart(df_plot, x="label", y=df_plot.columns[0])
            elif atype == "descriptive":
                st.dataframe(result.get("stats"))

            if result.get("summary"):
                st.subheader("Business Summary")
                st.write(result["summary"])

            rec_models = result.get("recommended_models")
            if rec_models:
                with st.expander("Recommended Models"):
                    st.json(rec_models)

            merge_report = result.get("model_info", {}).get("merge_report")
            if merge_report:
                with st.expander("Merge Report"):
                    st.download_button(
                        "Download merge_report.json",
                        json.dumps(merge_report, indent=2),
                        file_name="merge_report.json",
                        mime="application/json",
                    )
                    st.json(merge_report)

            explanations = result.get("model_info", {}).get("explanations")
            if explanations:
                fi = explanations.get("feature_importances") or explanations.get("coefficients")
                if fi is not None:
                    with st.expander("Feature Importances"):
                        st.bar_chart(pd.Series(fi))
                if explanations.get("shap_values") is not None:
                    with st.expander("SHAP Values"):
                        st.download_button(
                            "Download shap_values.json",
                            json.dumps(explanations["shap_values"], indent=2),
                            file_name="shap_values.json",
                            mime="application/json",
                        )

            # Model download (sidebar → now inline)
            model_path = Path("best_model.pkl")
            if model_path.exists():
                st.download_button(
                    "Download best_model.pkl",
                    model_path.read_bytes(),
                    file_name="best_model.pkl",
                    mime="application/octet-stream",
                )

            # HTML report
            try:
                report_bytes = generate_report_bytes(data, result, title="Assay Analysis Report")
                st.download_button(
                    "Export Full Report (HTML)",
                    report_bytes,
                    file_name="analysis_report.html",
                    mime="text/html",
                )
            except Exception:
                pass

            # Premium download
            dl_button = getattr(st.sidebar, "download_button", None)
            if callable(dl_button):
                dl_button(
                    "Download One Pager (Premium)",
                    result.get("summary", ""),
                    file_name="one_pager.txt",
                    disabled=not st.session_state.get("is_paid", False),
                )



# ═══════════════════════════════════
# DATA TAB — Upload + Explore
# ═══════════════════════════════════

with data_tab:
    # Dataset upload section
    with st.expander("Add Dataset", expanded=data is None):
        add_mode = st.radio("Source", ["Upload CSV", "Connect API", "Sample Data"], horizontal=True)

        if add_mode == "Upload CSV":
            uploaded_files = st.file_uploader("Upload CSV", type="csv", accept_multiple_files=True)
            if uploaded_files:
                for f in uploaded_files:
                    name = f.name
                    if name not in st.session_state["datasets"]:
                        with st.spinner(f"Loading {name}..."):
                            upload_dir = Path("User_Data/uploaded")
                            upload_dir.mkdir(parents=True, exist_ok=True)
                            file_path = upload_dir / name
                            with open(file_path, "wb") as buffer:
                                f.seek(0)
                                buffer.write(f.read())
                            from preprocessing.data_cleaning import standardize_dataframe
                            f.seek(0)
                            df = pd.read_csv(f)
                            df = standardize_dataframe(df)
                            st.session_state["datasets"][name] = df
                            _persist_dataset(name, df)
                            if "dataset_paths" not in st.session_state:
                                st.session_state["dataset_paths"] = {}
                            st.session_state["dataset_paths"][name] = str(file_path.absolute())
                            if not st.session_state["primary_dataset_id"]:
                                st.session_state["primary_dataset_id"] = name
                st.rerun()

        elif add_mode == "Connect API":
            st.caption("Load data directly from external sources")
            source = st.selectbox(
                "Source",
                ["FRED (Econ)", "World Bank", "Custom / Generic API"],
                key="data_tab_connect_source",
            )
            if source == "FRED (Econ)":
                try:
                    from public_data.connectors import FREDConnector
                    fred = FREDConnector()
                    series = [s.id for s in fred.get_available_series()]
                    selected_series = st.multiselect("Select Series", series, default=["GDP"], key="data_tab_fred_sel")
                    if st.button("Load FRED Data", key="data_tab_load_fred"):
                        with st.spinner("Fetching..."):
                            loaded_count = 0
                            for s_id in selected_series:
                                try:
                                    df = fred.fetch_data(s_id)
                                    if not df.empty:
                                        st.session_state["datasets"][f"FRED_{s_id}"] = df
                                        st.session_state["primary_dataset_id"] = f"FRED_{s_id}"
                                        _persist_dataset(f"FRED_{s_id}", df)
                                        loaded_count += 1
                                except Exception as e:
                                    st.error(f"Could not load {s_id}: {e}")
                            if loaded_count > 0:
                                st.success(f"Loaded {loaded_count} datasets")
                                st.rerun()
                except ImportError:
                    st.error("FRED connector missing")

            elif source == "World Bank":
                try:
                    from public_data.connectors import WorldBankConnector
                    wb = WorldBankConnector()
                    indicators = [s.id for s in wb.get_available_series()]
                    selected_inds = st.multiselect("Select Indicators", indicators, default=["NY.GDP.MKTP.KD.ZG"], key="data_tab_wb_sel")
                    country = st.text_input("Country Code", "USA", key="data_tab_wb_country")
                    if st.button("Load World Bank Data", key="data_tab_load_wb"):
                        with st.spinner("Fetching..."):
                            loaded_count = 0
                            for ind in selected_inds:
                                try:
                                    new_df = wb.fetch_data(ind, countries=country)
                                    if not new_df.empty:
                                        name = f"WB_{ind}_{country}"
                                        st.session_state["datasets"][name] = new_df
                                        st.session_state["primary_dataset_id"] = name
                                        loaded_count += 1
                                except Exception as e:
                                    st.error(f"Could not retrieve {ind}: {e}")
                            if loaded_count > 0:
                                st.success(f"Loaded {loaded_count} datasets")
                                st.rerun()
                except ImportError:
                    st.error("World Bank connector missing")

            elif source == "Custom / Generic API":
                api_url = st.text_input("API URL", "https://api.coindesk.com/v1/bpi/currentprice.json")
                method = st.selectbox("Method", ["GET", "POST"])
                headers_str = st.text_area("Headers (JSON)", "{}")
                data_key = st.text_input("Data Key", value=None)
                if st.button("Fetch"):
                    try:
                        import requests
                        headers = json.loads(headers_str)
                        with st.spinner(f"Fetching {api_url}..."):
                            resp = requests.get(api_url, headers=headers) if method == "GET" else requests.post(api_url, headers=headers)
                            resp.raise_for_status()
                            data_json = resp.json()
                            if data_key and data_key in data_json:
                                data_json = data_json[data_key]
                            if isinstance(data_json, dict):
                                df = pd.json_normalize(data_json)
                            elif isinstance(data_json, list):
                                df = pd.DataFrame(data_json)
                            else:
                                st.error("Could not parse response into DataFrame")
                                df = pd.DataFrame()
                            if not df.empty:
                                from preprocessing.data_cleaning import standardize_dataframe
                                df = standardize_dataframe(df)
                                st.session_state["datasets"]["API_Custom_Data"] = df
                                st.session_state["primary_dataset_id"] = "API_Custom_Data"
                                st.success("Loaded Custom Data")
                                st.rerun()
                    except Exception as e:
                        st.error(f"API request failed: {e}")

        elif add_mode == "Sample Data":
            from preprocessing.data_cleaning import standardize_dataframe
            if st.button("Load Bitcoin Sample"):
                try:
                    df = pd.read_csv("data/bitcoin_history.csv")
                    df = standardize_dataframe(df)
                    st.session_state["datasets"]["Bitcoin"] = df
                    if not st.session_state["primary_dataset_id"]:
                        st.session_state["primary_dataset_id"] = "Bitcoin"
                    st.rerun()
                except Exception:
                    st.error("Sample file missing")
                    st.session_state["datasets"]["Mock Sales"] = standardize_dataframe(mock_data)
                    st.session_state["primary_dataset_id"] = "Mock Sales"
                    st.rerun()

    # Dataset selector (if multiple)
    if len(st.session_state["datasets"]) > 1:
        active_ds = st.radio(
            "Active dataset",
            options=list(st.session_state["datasets"].keys()),
            index=list(st.session_state["datasets"].keys()).index(
                st.session_state["primary_dataset_id"]
            ) if st.session_state["primary_dataset_id"] in st.session_state["datasets"] else 0,
            horizontal=True,
        )
        if active_ds != st.session_state["primary_dataset_id"]:
            st.session_state["primary_dataset_id"] = active_ds
            st.rerun()

    if data is not None:
        sub_preview, sub_enrich, sub_lineage = st.tabs(
            ["Preview", "Enrich", "Lineage"]
        )

        with sub_preview:
            # Format datetime columns to YYYY-MM-DD to save space and reduce noise
            display_preview = preview.copy()
            for col in display_preview.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns:
                display_preview[col] = display_preview[col].dt.strftime('%Y-%m-%d')
            st.dataframe(display_preview)
        with sub_enrich:
            try:
                render_action_center(data, meta)
            except Exception as e:
                st.error(f"Enrich tab error: {e}")

        with sub_lineage:
            try:
                from ui.data_fabric import render_data_fabric_tab
                meta_dicts = [
                    {"original_name": m.name, "semantic_type": getattr(m, 'semantic_type', None)}
                    for m in meta
                ] if meta else []
                render_data_fabric_tab(data, meta_dicts)
            except ImportError as e:
                st.warning(f"Data Fabric module not available: {e}")
            except Exception as e:
                st.error(f"Lineage tab error: {e}")

# ═══════════════════════════════════
# EXPLORE TAB
# ═══════════════════════════════════

with explore_tab:
    if data is None:
        st.info("Load a dataset in the Data tab first to explore it.")
    else:
        try:
            dataset_id = st.session_state.get("primary_dataset_id")
            current_meta = st.session_state.get("dataset_metadata", {}).get(dataset_id, {})
            context_parts = []
            if dataset_id:
                context_parts.append(f"Dataset name: {dataset_id}")
            if current_meta.get("context"):
                context_parts.append(current_meta.get("context"))
            elif current_meta.get("description"):
                context_parts.append(current_meta.get("description"))
            context_str = ". ".join(context_parts)
            render_exploratory_tab(data, context=context_str)
        except Exception as e:
            st.error(f"Explore tab error: {e}")

# ═══════════════════════════════════
# SETTINGS TAB
# ═══════════════════════════════════

with settings_tab:
    sub_llm, sub_agents, sub_scheduler, sub_dev = st.tabs(
        ["LLM", "Agents", "Scheduler", "Dev"]
    )

    with sub_llm:
        render_llm_settings()
        st.divider()
        try:
            from ui.learning_progress import render_learning_progress
            render_learning_progress(compact=False)
        except ImportError:
            pass
        st.divider()
        try:
            from ui.teach_logic import render_teaching_mode
            render_teaching_mode()
        except ImportError:
            st.info("Teaching module not loaded.")

    with sub_agents:
        try:
            from ui.agent_control import render_agent_control
            render_agent_control()
        except ImportError:
            st.info("Agent control module not loaded.")
        except Exception as e:
            st.error(f"Agent control error: {e}")

    with sub_scheduler:
        try:
            from ui.scheduler_ui import render_scheduler_ui
            render_scheduler_ui()
        except ImportError:
            st.warning("Scheduler module not found.")
        except Exception as e:
            st.error(f"Scheduler error: {e}")

    with sub_dev:
        try:
            from ui.dev_settings import render_dev_toggle
            render_dev_toggle()
        except ImportError:
            pass
        try:
            from ui.diagnostics import render_diagnostics
            render_diagnostics()
        except ImportError:
            pass
        except Exception as e:
            st.caption(f"Diagnostics unavailable: {e}")
