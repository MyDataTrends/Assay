"""
Natural Language Query UI - Ask questions about your data in plain English.

Provides a chat-like interface where users can:
- Ask questions about their data
- Request specific analyses or calculations
- Generate visualizations from descriptions
"""
import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
import json
import traceback


def _get_llm_response(prompt: str, max_tokens: int = 1024) -> str:
    """Get LLM response using the unified interface."""
    try:
        from llm_manager.llm_interface import get_llm_completion, is_llm_available
        if is_llm_available():
            return get_llm_completion(prompt, max_tokens=max_tokens, temperature=0.3)
    except ImportError:
        pass
    
    # Fallback disabled - causes C-level crashes
    return ""



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
        # Silent failure for dashboard components
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
    except Exception:
        return ""


def _generate_code_for_query(df: pd.DataFrame, query: str) -> str:
    """Generate pandas code to answer a natural language query."""
    columns = df.columns.tolist()
    dtypes = {col: str(df[col].dtype) for col in columns}
    sample = df.head(3).to_dict()
    
    # RAG Injection
    rag_context = get_rag_context(query)
    
    prompt = f"""You are a data analyst. Generate Python pandas code to answer this question.

DATAFRAME INFO:
- Variable name: df
- Columns: {columns}
- Types: {dtypes}
- Sample: {sample}

QUESTION: {query}
{rag_context}

Generate ONLY the Python code (no markdown, no explanation). The code should:
1. Use the 'df' variable
2. Store the result in a variable called 'result'
3. Be safe (no file operations, no imports except pandas/numpy)

Code:"""
    
    return _get_llm_response(prompt, max_tokens=500)


def _generate_natural_answer(df: pd.DataFrame, query: str, result: Any) -> str:
    """Generate a natural language answer from code result."""
    prompt = f"""Based on this data analysis result, provide a brief natural language answer.

QUESTION: {query}
RESULT: {result}

Answer in 1-2 sentences:"""
    
    return _get_llm_response(prompt, max_tokens=200)


def _safe_execute_code(code: str, df: pd.DataFrame) -> tuple:
    """
    Safely execute generated code.
    
    Returns (success, result, error)
    """
    import numpy as np
    
    # Clean up the code
    code = code.strip()
    
    # Remove markdown code blocks if present
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    
    # Security checks
    dangerous_patterns = [
        "import os", "import sys", "import subprocess",
        "open(", "exec(", "eval(", "__import__",
        "rm ", "del ", "shutil", "requests.",
    ]
    
    for pattern in dangerous_patterns:
        if pattern in code:
            return False, None, f"Blocked unsafe pattern: {pattern}"
    
    # Execute in restricted namespace
    namespace = {
        "df": df,
        "pd": pd,
        "np": np,
        "result": None,
    }
    
    try:
        exec(code, namespace)
        result = namespace.get("result")
        return True, result, None
    except Exception as e:
        return False, None, str(e)


def render_nl_query_panel(df: pd.DataFrame):
    """Render the natural language query panel."""
    
    st.subheader("💬 Ask Your Data")
    st.caption("Ask questions about your data in plain English")
    
    # Check LLM availability
    llm_available = False
    try:
        from llm_manager.llm_interface import is_llm_available
        llm_available = is_llm_available()
    except ImportError:
        pass
    
    if not llm_available:
        st.warning("⚠️ No LLM configured. Go to '🤖 LLM Settings' tab to set up a model.")
        return
    
    # Query history in session state
    if "nl_query_history" not in st.session_state:
        st.session_state["nl_query_history"] = []
    
    # Example queries
    with st.expander("📝 Example queries", expanded=False):
        examples = [
            "What is the average value of [column]?",
            "Show me the top 10 rows by [column]",
            "How many unique values are in [column]?",
            "What is the correlation between [col1] and [col2]?",
            "Group by [column] and sum [value_column]",
            "What is the distribution of [column]?",
            "Find rows where [column] > [value]",
        ]
        for ex in examples:
            st.caption(f"• {ex}")
    
    # Query input
    query = st.text_input(
        "Your question:",
        placeholder="e.g., What is the average sales by region?",
        key="nl_query_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_query = st.button("🔍 Ask", type="primary")
    with col2:
        show_code = st.checkbox("Show generated code", value=False)
    
    if run_query and query:
        with st.spinner("Analyzing your question..."):
            # Generate code
            generated_code = _generate_code_for_query(df, query)
            
            if not generated_code:
                st.error("Failed to generate analysis code. Please try rephrasing your question.")
                return
            
            # Show code if requested
            if show_code:
                with st.expander("Generated Code", expanded=True):
                    st.code(generated_code, language="python")
            
            # Execute code
            success, result, error = _safe_execute_code(generated_code, df)
            
            if success:
                # Implicit Logging
                _, _, logger = init_learning_system()
                if logger:
                    logger.log_interaction(
                        messages=[{"role": "user", "content": query}, {"role": "assistant", "content": generated_code}],
                        metadata={"type": "analysis", "code": generated_code},
                        success=True
                    )
                
                # Generate natural answer
                natural_answer = _generate_natural_answer(df, query, result)
                
                if natural_answer:
                    st.success(natural_answer)
                
                # Display result
                st.markdown("**Result:**")
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                elif isinstance(result, pd.Series):
                    st.dataframe(result.to_frame())
                elif isinstance(result, (dict, list)):
                    st.json(result)
                else:
                    st.write(result)
                
                # Add to history
                st.session_state["nl_query_history"].append({
                    "query": query,
                    "answer": natural_answer,
                    "result": str(result)[:200],
                })
            else:
                st.error(f"Error executing analysis: {error}")
                if show_code:
                    st.info("Try rephrasing your question or check the generated code above.")
    
    # Show history
    if st.session_state["nl_query_history"]:
        with st.expander("📜 Query History", expanded=False):
            for i, item in enumerate(reversed(st.session_state["nl_query_history"][-5:])):
                st.markdown(f"**Q:** {item['query']}")
                st.caption(item['answer'])
                st.divider()


def render_chart_from_description(df: pd.DataFrame):
    """Generate a chart from a natural language description."""
    
    st.subheader("📊 Describe a Chart")
    st.caption("Describe the visualization you want and let AI create it")
    
    # Check LLM
    llm_available = False
    try:
        from llm_manager.llm_interface import is_llm_available
        llm_available = is_llm_available()
    except ImportError:
        pass
    
    if not llm_available:
        st.warning("⚠️ No LLM configured. Go to '🤖 LLM Settings' tab to set up a model.")
        return
    
    description = st.text_area(
        "Describe your chart:",
        placeholder="e.g., A bar chart showing average sales by category, sorted descending",
        key="chart_description",
        height=100
    )
    
    if st.button("🎨 Generate Chart", disabled=not description):
        with st.spinner("Creating your visualization..."):
            columns = df.columns.tolist()
            dtypes = {col: str(df[col].dtype) for col in columns}
            
            # RAG Injection for charts
            rag_context = get_rag_context(description)
            
            prompt = f"""Generate Python code to create a Plotly chart based on this description.

DATAFRAME (df):
- Columns: {columns}
- Types: {dtypes}

DESCRIPTION: {description}
{rag_context}

Generate ONLY Python code that:
1. Uses 'df' variable
2. Uses plotly.express as 'px'
3. Stores the figure in 'fig'
4. Uses template="plotly_dark"

Code:"""
            
            code = _get_llm_response(prompt, max_tokens=600)
            
            # Validate that we got actual code
            if not code or len(code) < 10:
                st.error("Failed to generate chart code. LLM returned empty response.")
                return
            
            # Check for error responses
            if "unavailable" in code.lower() or "error" in code.lower()[:20]:
                st.error("LLM is not available. Please configure a model in '🤖 LLM Settings' tab.")
                return
            
            # Show code
            with st.expander("Generated Code"):
                st.code(code, language="python")
            
            # Clean code
            code = code.strip()
            if code.startswith("```"):
                lines = code.split("\n")
                # Remove first line (```python) and last line (```)
                if lines[-1].strip() == "```":
                    code = "\n".join(lines[1:-1])
                else:
                    code = "\n".join(lines[1:])
            
            # Additional validation - must contain 'fig' and 'px'
            if "fig" not in code or "px." not in code:
                st.error("Generated code doesn't look like valid Plotly code. Please try a different description.")
                return
            
            # Execute
            try:
                import plotly.express as px
                import numpy as np
                
                namespace = {"df": df, "px": px, "pd": pd, "np": np, "fig": None}
                exec(code, namespace)
                
                if namespace.get("fig"):
                    st.plotly_chart(namespace["fig"], width="stretch",
                                     key=f"nlchart_{hash(description)}")
                    
                    # Implicit Logging
                    _, _, logger = init_learning_system()
                    if logger:
                        logger.log_interaction(
                            messages=[{"role": "user", "content": description}, {"role": "assistant", "content": code}],
                            metadata={"type": "visualization", "code": code},
                            success=True
                        )
                else:
                    st.error("Chart generation failed - no figure created")
            except SyntaxError as e:
                st.error(f"Syntax error in generated code: {e}")
                st.info("Try rephrasing your chart description.")
            except Exception as e:
                st.error(f"Error creating chart: {e}")

