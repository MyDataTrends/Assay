"""
Chat Mode - Pure conversation interface for Assay.

A Julius AI-style chat experience where users can:
- Upload data via chat
- Ask questions in natural language
- Get visualizations and insights through conversation
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import sys
from utils.logging import get_logger

logger = get_logger(__name__)

# Ensure project root is in path
# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================================
# LLM Integration
# ============================================================================

def get_llm_response(prompt: str, max_tokens: int = 1024) -> str:
    """Get LLM response using the unified interface."""
    try:
        from llm_manager.llm_interface import get_llm_completion, is_llm_available
        if is_llm_available():
            return get_llm_completion(prompt, max_tokens=max_tokens, temperature=0.3)
    except ImportError:
        pass
    except Exception as e:
        print(f"LLM Error: {e}")
    return ""


def is_llm_ready() -> bool:
    """Check if LLM is available."""
    try:
        from llm_manager.llm_interface import is_llm_available
        return is_llm_available()
    except ImportError:
        return False


# ============================================================================
# Code Generation & Execution
# ============================================================================


# ============================================================================
# Cascade Planner Integration (2026 Execution Hardening)
# ============================================================================

def cascade_detect_intent(query: str) -> tuple:
    """
    Detect intent using the cascade planner's deterministic rules.
    
    Returns:
        Tuple of (intent_name, confidence, cascade_intent_enum)
    """
    try:
        from orchestration import classify_intent, Intent
        
        intent, confidence = classify_intent(query)
        
        # Map cascade intents to legacy chat_logic intents
        intent_mapping = {
            Intent.DESCRIBE_DATA: "informational",
            Intent.VISUALIZE: "visualization",
            Intent.TRANSFORM: "analysis",
            Intent.FILTER: "analysis",
            Intent.AGGREGATE: "analysis",
            Intent.MODEL_TRAIN: "analysis",
            Intent.MODEL_PREDICT: "analysis",
            Intent.ENRICH_DATA: "analysis",
            Intent.EXPORT: "analysis",
            Intent.COMPARE: "analysis",
            Intent.UNKNOWN: "analysis",
        }
        
        legacy_intent = intent_mapping.get(intent, "analysis")
        return legacy_intent, confidence, intent
        
    except ImportError:
        return "analysis", 0.5, None
    except Exception as e:
        print(f"Cascade intent detection failed: {e}")
        return "analysis", 0.5, None


def cascade_execute(df, query: str, context: dict = None) -> dict:
    """
    Execute a query using the cascade planner.
    
    This is the structured alternative to raw code generation + exec().
    Uses registered tools with retry logic and fallbacks.
    
    Returns:
        Dict with keys: success, output, error, intent, steps_completed
    """
    try:
        from orchestration import get_planner, get_artifact_store
        
        context = context or {}
        context["df"] = df
        
        planner = get_planner()
        store = get_artifact_store()
        
        # Generate and execute plan
        plan = planner.plan(query, context=context)
        result = planner.execute(plan, context=context)
        
        # Save artifact for replay
        try:
            store.save(plan, result, context)
        except Exception as e:
            print(f"Artifact save failed: {e}")
        
        # Determine plan type from confidence level
        confidence = plan.metadata.get("confidence", 0.0)
        if confidence >= 0.7:
            plan_type = "template"
        elif confidence >= 0.6:
            plan_type = "llm_generated"
        else:
            plan_type = "learned"

        steps = [
            {
                "step_num": i + 1,
                "label": step.action,
                "tool": step.tool,
                "status": step.status.value if hasattr(step.status, "value") else str(step.status),
                "duration_ms": step.execution_time_ms,
                "error": step.error,
                "output_preview": str(step.result)[:200] if step.result is not None else None,
            }
            for i, step in enumerate(plan.steps)
        ]

        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "intent": plan.intent.value,
            "plan_id": plan.plan_id,
            "steps_completed": result.steps_completed,
            "total_steps": result.total_steps,
            "plan_type": plan_type,
            "confidence": confidence,
            "steps": steps,
        }
        
    except ImportError as e:
        print(f"Orchestration not available: {e}")
        return {"success": False, "error": "Orchestration not available", "output": None}
    except Exception as e:
        print(f"Cascade execution failed: {e}")
        return {"success": False, "error": str(e), "output": None}


def should_use_cascade(query: str) -> bool:
    """
    Determine if query should use cascade planner vs legacy code generation.
    
    Cascade is preferred for:
    - Data profiling/description
    - Simple visualizations
    - Standard transformations (filter, group, aggregate)
    
    Legacy code generation is preferred for:
    - Complex custom analysis
    - Unusual visualizations
    - Multi-step custom workflows
    """
    try:
        from orchestration import classify_intent, Intent
        
        intent, confidence = classify_intent(query)
        
        # High-confidence known intents use cascade
        cascade_intents = {
            Intent.DESCRIBE_DATA,
            Intent.VISUALIZE,
            Intent.FILTER,
            Intent.AGGREGATE,
        }
        
        if intent in cascade_intents and confidence >= 0.8:
            return True
            
        # Fallback keyword detection for visualization if confidence is low
        # This catches "plot", "chart", "graph" even if classifier is unsure
        if "visual" in query.lower() or "chart" in query.lower() or "plot" in query.lower() or "graph" in query.lower():
            if intent != Intent.VISUALIZE:
                # Force visualization intent if keywords are strong
                return True
        
        return False
        
    except ImportError:
        return False


# ============================================================================
# Learning System Integration
# ============================================================================

@st.cache_resource
def init_learning_system():
    """Initialize vector store and logger."""
    try:
        from learning.vector_store import VectorStore
        from learning.embeddings import EmbeddingModel
        from llm_learning.interaction_logger import get_interaction_logger
        
        vs = VectorStore()
        emb = EmbeddingModel()
        logger = get_interaction_logger()
        return vs, emb, logger
    except Exception as e:
        print(f"Learning system init failed: {e}")
        return None, None, None

def get_rag_context(query: str) -> str:
    """Retrieve similar code examples for the query."""
    vs, emb, _ = init_learning_system()
    if not vs or not emb:
        return ""
        
    try:
        vector = emb.embed_query(query)
        results = vs.search(vector, limit=2, threshold=0.7)
        if not results:
            return ""
            
        examples_str = "\nRELEVANT EXAMPLES:\n"
        for res in results:
            examples_str += f"- Intent: {res['intent']}\n  Code:\n{res['code']}\n"
        return examples_str
    except Exception as e:
        print(f"RAG retrieval failed: {e}")
        return ""

def get_style_rules() -> str:
    """Retrieve active style rules from the vector store."""
    vs, emb, _ = init_learning_system()
    if not vs:
        return ""
    
    # We cheat a bit: we search for "style formatting rules" to get relevant ones,
    # or we could just query for source='style_rule' if we added a specific method for that.
    # For now, let's use a broad query to get the top style guides.
    try:
        # Check if we can filter by source in a raw query (faster/better)
        # But stick to the public API for safety: search for "formatting style guidelines"
        if emb:
            vector = emb.embed_query("formatting style guidelines colors fonts")
            results = vs.search(vector, limit=5, threshold=0.4) 
            
            # Filter client-side for source='style_rule' to be sure
            style_rules = [res for res in results if res['source'] == 'style_rule']
            
            if not style_rules:
                return ""
                
            rules_str = "\nSTYLE & FORMATTING GUIDELINES:\n"
            for res in style_rules:
                rules_str += f"- {res['intent']}: {res['code']}\n"
            return rules_str
            
    except Exception as e:
        print(f"Style retrieval failed: {e}")
        
    return ""

def generate_analysis_code(df: pd.DataFrame, query: str, context: str = "", datasets: Optional[Dict[str, pd.DataFrame]] = None) -> str:
    """Generate pandas code to answer a natural language query."""
    columns = df.columns.tolist()
    dtypes = {col: str(df[col].dtype) for col in columns}
    sample = df.head(3).to_string()
    
    context_str = f"\nDATASET CONTEXT: {context}\n" if context else ""
    
    # RAG Injection
    rag_context = get_rag_context(query)
    style_context = get_style_rules()
    
    # Add datasets info if available
    datasets_info = ""
    if datasets:
        datasets_info = "\nAVAILABLE DATASETS (access via 'datasets' dict):\n"
        for name in datasets:
            if name != "primary": # avoid confusion if primary is in there twice
                datasets_info += f"- datasets['{name}']\n"

    prompt = f"""You are a data analyst. Generate Python pandas code to answer this question.
{context_str}
{rag_context}
{style_context}
{datasets_info}
DATAFRAME INFO:
- Variable name: df (Primary Dataset)
- Columns: {columns}
- Types: {dtypes}
- Sample data:
{sample}

QUESTION: {query}

Generate ONLY the Python code (no markdown, no explanation). The code must be valid Python.
1. Use the 'df' variable for the primary dataset.
2. Use 'datasets[name]' for other datasets if needed.
3. Store the final result in a variable called 'result'.
4. Be safe (no file operations, no external imports).
5. DO NOT use multi-line f-strings with single quotes. Use concatenation or triple quotes if needed.
6. DO NOT generate plots or charts here.

Code:"""
    
    return get_llm_response(prompt, max_tokens=500)


def generate_visualization_code(df: pd.DataFrame, query: str, context: str = "", datasets: Optional[Dict[str, pd.DataFrame]] = None) -> str:
    """Generate Plotly code for a visualization request."""
    columns = df.columns.tolist()
    dtypes = {col: str(df[col].dtype) for col in columns}
    sample = df.head(3).to_string()
    
    context_str = f"\nDATASET CONTEXT: {context}\n" if context else ""
    
    # RAG Injection
    rag_context = get_rag_context(query)
    
    # Add datasets info if available
    datasets_info = ""
    if datasets:
        datasets_info = "\nAVAILABLE DATASETS (access via 'datasets' dict):\n"
        for name in datasets:
             datasets_info += f"- datasets['{name}']\n"

    prompt = f"""Generate Python code to create a Plotly visualization.
{context_str}
{datasets_info}
DATAFRAME (df):
- Columns: {columns}
- Types: {dtypes}
- Sample data:
{sample}

REQUEST: {query}

Generate ONLY Python code to create a Plotly visualization.
1. The code must be valid Python.
2. Use 'df' for the primary dataset (or df = datasets['name'] if referring to others).
3. Use plotly.express as 'px'.
4. Store the figure in a variable named 'fig'.
5. Use template="plotly_dark".
6. Title the chart appropriately based on the data.
7. DO NOT use multi-line f-strings.
8. Handle potential missing values if necessary.

Code:"""
    
    return get_llm_response(prompt, max_tokens=600)


def safe_execute(code: str, df: pd.DataFrame, datasets: Optional[Dict[str, pd.DataFrame]] = None) -> tuple:
    """
    Safely execute generated code.
    Returns (success, result, error)
    """
    # Clean code
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    
    # Security checks
    dangerous = ["import os", "import sys", "subprocess", "open(", "exec(", 
                 "eval(", "__import__", "shutil", "requests."]
    for pattern in dangerous:
        if pattern in code:
            logger.warning(f"Security: Blocked code containing '{pattern}'")
            return False, None, f"Blocked unsafe pattern: {pattern}"
    
    # Execute
    logger.info("Security: Executing generated code (Blocklist Sandbox)")
    namespace = {"df": df, "pd": pd, "np": np, "result": None}
    if datasets:
        namespace["datasets"] = datasets # Inject all datasets
        
    try:
        exec(code, namespace)
        return True, namespace.get("result"), None
    except Exception as e:
        return False, None, str(e)


def safe_execute_viz(code: str, df: pd.DataFrame, datasets: Optional[Dict[str, pd.DataFrame]] = None) -> tuple:
    """Execute visualization code. Returns (success, fig, error)"""
    import plotly.express as px
    
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    
    namespace = {"df": df, "pd": pd, "np": np, "px": px, "fig": None}
    if datasets:
        namespace["datasets"] = datasets
        
    try:
        exec(code, namespace)
        return True, namespace.get("fig"), None
    except Exception as e:
        return False, None, str(e)


def fallback_visualization(df: pd.DataFrame, query: str):
    """
    Create a reasonable visualization without LLM based on query keywords and data structure.
    Returns a Plotly figure or None.
    """
    import plotly.express as px
    
    query_lower = query.lower()
    num_cols = df.select_dtypes(include=['int64', 'float64', 'float32', 'int32']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not num_cols:
        return None
    
    # Try to identify what column they're asking about
    mentioned_cols = []
    for col in df.columns:
        if col.lower() in query_lower:
            mentioned_cols.append(col)
    
    try:
        # "highest" / "top" / "best" -> bar chart sorted descending
        if any(kw in query_lower for kw in ["highest", "top", "best", "most", "largest"]):
            if cat_cols and num_cols:
                # Group by first cat column, sum/mean first num column
                agg_col = mentioned_cols[0] if mentioned_cols and mentioned_cols[0] in num_cols else num_cols[0]
                group_col = cat_cols[0]
                grouped = df.groupby(group_col)[agg_col].sum().nlargest(10).reset_index()
                fig = px.bar(grouped, x=group_col, y=agg_col, 
                           title=f"Top 10 {group_col} by {agg_col}",
                           template="plotly_dark")
                return fig
            elif len(num_cols) >= 2:
                # Show top rows by first numeric column
                top_df = df.nlargest(10, num_cols[0])
                fig = px.bar(top_df, y=num_cols[0], title=f"Top 10 by {num_cols[0]}", 
                           template="plotly_dark")
                return fig
        
        # "trend" / "over time" / "line" -> line chart
        if any(kw in query_lower for kw in ["trend", "over time", "line", "time"]):
            y_col = mentioned_cols[0] if mentioned_cols and mentioned_cols[0] in num_cols else num_cols[0]
            fig = px.line(df.head(100), y=y_col, title=f"{y_col} Trend", template="plotly_dark")
            return fig
        
        # "distribution" / "histogram" -> histogram
        if any(kw in query_lower for kw in ["distribution", "histogram", "spread"]):
            col = mentioned_cols[0] if mentioned_cols and mentioned_cols[0] in num_cols else num_cols[0]
            fig = px.histogram(df, x=col, title=f"Distribution of {col}", template="plotly_dark")
            return fig
        
        # Default: bar chart of aggregated data or line of first numeric column
        if cat_cols and num_cols:
            grouped = df.groupby(cat_cols[0])[num_cols[0]].sum().nlargest(10).reset_index()
            fig = px.bar(grouped, x=cat_cols[0], y=num_cols[0],
                       title=f"{num_cols[0]} by {cat_cols[0]}", template="plotly_dark")
            return fig
        else:
            fig = px.line(df.head(100), y=num_cols[0], title=f"{num_cols[0]}", template="plotly_dark")
            return fig
            
    except Exception as e:
        return None


def execute_analysis_with_retry(df: pd.DataFrame, query: str, context: str = "", datasets: Optional[Dict[str, pd.DataFrame]] = None, max_retries: int = 3) -> tuple:
    """
    Generate and execute analysis code with self-correction retry loop.
    Returns (success, result, code, error_message)
    """
    import time
    
    # Initial generation
    code = generate_analysis_code(df, query, context, datasets)
    
    for attempt in range(max_retries + 1):
        if not code:
            return False, None, "", "LLM failed to generate code."
            
        success, result, error = safe_execute(code, df, datasets)
        
        if success:
            return True, result, code, None
            
        # If failed, try to correct
        if attempt < max_retries:
            print(f"Attempt {attempt+1} failed: {error}. Retrying...")
            
            # Short sleep to avoid rate limits
            time.sleep(1)
            
            correction_prompt = f"""The previous Python code you generated failed with this error:
{error}

Original Question: {query}

Generated Code:
{code}

Please correct the code to fix the error. Return ONLY the corrected Python code.
CRITICAL FIX INSTRUCTIONS:
1. Fix the specific error mentioned above.
2. Check for unclosed parentheses ')' or brackets ']'.
3. Check for unclosed string literals (match your quotes).
4. Avoid multi-line f-strings (they are error-prone).
5. Ensure all lines are properly indented.
"""
            
            code = get_llm_response(correction_prompt, max_tokens=600)
            
    return False, None, code, error


# ============================================================================
# Intent Detection
# ============================================================================


def extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON object from LLM response text."""
    try:
        # Strategy 1: Direct JSON parse
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    try:
        # Strategy 2: Extract JSON from code block or substrings
        import re
        # Look for { ... } structure
        matches = re.findall(r'\{[^{}]+\}', response, re.DOTALL)
        if matches:
            # Try the first one that parses
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # Look for ```json ... ``` blocks
        if "```" in response:
            lines = response.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if "```" in line:
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            if json_lines:
                return json.loads("\n".join(json_lines))
                
    except Exception:
        pass
        
    return None

def detect_intent(query: str, context: str = "") -> str:
    """
    Detect intent using Smart Routing.
    
    Priority:
    1. Cascade planner's deterministic rules (fast, no LLM)
    2. Fast keyword checks
    3. LLM Smart Routing
    4. Keyword fallback
    """
    query_lower = query.lower()
    
    # 0. Fast Keyword Check (Optimization for obvious cases)
    # "Show me..." is almost always a chart
    if query_lower.startswith("show me") or any(k in query_lower for k in ["plot", "chart", "graph", "visualization", "visualise", "scatter", "histogram"]):
        return "visualization"
        
    # "Describe/Explain" is almost always text
    if any(k in query_lower for k in ["describe", "explain", "summarize", "tell me about"]):
        return "informational"

    # 1. Try Cascade Planner's deterministic detection first (fast, no LLM call)
    try:
        legacy_intent, confidence, cascade_intent = cascade_detect_intent(query)
        if confidence >= 0.8:
            print(f"[Cascade] Intent: {cascade_intent.value if cascade_intent else legacy_intent} (conf: {confidence:.2f})")
            return legacy_intent
    except Exception as e:
        print(f"Cascade detection skipped: {e}")
        
    # 2. LLM Smart Routing
    try:
        prompt = f"""Classify the user's intent into one of these categories:
1. "visualization": The user wants to see a chart, graph, or plot.
2. "analysis": The user wants a calculation, transformation, or data manipulation (e.g. "average", "filter", "find").
3. "informational": The user is asking a general question about the dataset or wants a text explanation (e.g. "what is this data?", "explain column X").

QUERY: {query}
CONTEXT: {context[:500] if context else "None"}

Return ONLY a JSON object: {{"intent": "visualization" | "analysis" | "informational", "confidence": 0.9}}"""
        
        response = get_llm_response(prompt, max_tokens=100)
        data = extract_json_from_response(response)
        
        if data and "intent" in data:
            return data["intent"]
            
    except Exception as e:
        print(f"LLM Routing failed: {e}")
    
    # 3. Keyword Fallback (Legacy Logic)
    viz_keywords = ["chart", "plot", "graph", "visualize", "display", 
                    "histogram", "scatter", "bar", "line", "pie", "trend"]
    info_keywords = ["explain", "describe", "what is", "meaning", "define", "summary", "summarize"]
    
    for kw in viz_keywords:
        if kw in query_lower:
            return "visualization"
            
    for kw in info_keywords:
        if kw in query_lower:
            return "informational"
    
    return "analysis"


def generate_informational_response(query: str, context: str = "") -> str:
    """Generate a purely natural language response using available context."""
    from llm_manager.llm_interface import get_llm_chat, is_llm_available
    
    if not is_llm_available():
        return ""

    system_prompt = f"""You are a helpful data analyst. Answer the user's question based on the dataset context provided.
    
CONTEXT:
{context}

Answer in a clear, helpful, and professional manner. Do not generate code."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    return get_llm_chat(messages, max_tokens=4096)


def generate_natural_answer(query: str, result: Any) -> str:
    """Generate a natural language explanation of the result."""
    prompt = f"""Based on this data analysis result, provide a brief natural language answer.

QUESTION: {query}
RESULT: {result}

Answer in 1-2 clear sentences:"""
    
    return get_llm_response(prompt, max_tokens=200)


# ============================================================================
# Chat Message Components
# ============================================================================

def render_message(role: str, content: str, data: Any = None):
    """Render a chat message."""
    with st.chat_message(role):
        st.markdown(content)
        if data is not None:
            if isinstance(data, pd.DataFrame):
                st.dataframe(data, width="stretch")
            elif hasattr(data, 'show'):  # Plotly figure
                st.plotly_chart(data, width="stretch")
            elif isinstance(data, (dict, list)):
                st.json(data)
            else:
                st.write(data)


def get_data_summary(df: pd.DataFrame) -> str:
    """Get a natural language summary of the DataFrame."""
    rows, cols = df.shape
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    summary = f"Your dataset has **{rows:,} rows** and **{cols} columns**.\n\n"
    
    if num_cols:
        summary += f"📊 **Numeric columns** ({len(num_cols)}): {', '.join(num_cols[:5])}"
        if len(num_cols) > 5:
            summary += f" +{len(num_cols)-5} more"
        summary += "\n\n"
    
    if cat_cols:
        summary += f"🏷️ **Categorical columns** ({len(cat_cols)}): {', '.join(cat_cols[:5])}"
        if len(cat_cols) > 5:
            summary += f" +{len(cat_cols)-5} more"
        summary += "\n\n"
    
    if date_cols:
        summary += f"📅 **Date columns** ({len(date_cols)}): {', '.join(date_cols)}\n\n"
    
    # Sample questions
    summary += "**Try asking:**\n"
    if num_cols:
        summary += f"- What's the average {num_cols[0]}?\n"
    if num_cols and cat_cols:
        summary += f"- Show me {num_cols[0]} by {cat_cols[0]}\n"
    if len(num_cols) >= 2:
        summary += f"- Is there a correlation between {num_cols[0]} and {num_cols[1]}?\n"
    
    return summary


# ============================================================================
# Main Chat Interface
# ============================================================================
# Logic module - no main execution block needed
