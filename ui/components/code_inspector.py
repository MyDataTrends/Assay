"""
Code Inspector Component.

"Show your work" expandable section.
Renders Python, SQL, and model details in a tabbed expander with copy/download.
"""
import streamlit as st
from typing import Optional


def render_code_inspector(
    code: Optional[str] = None,
    sql: Optional[str] = None,
    model_details: Optional[dict] = None,
    label: str = "Show your work",
):
    """
    Render an expandable inspector showing what the system actually ran.

    Args:
        code: Python code string.
        sql: SQL query string.
        model_details: Dict with keys like model_type, target, inputs, assumptions.
        label: Expander label.
    """
    if not code and not sql and not model_details:
        return

    with st.expander(f"🔍 {label}", expanded=False):
        tabs = []
        tab_labels = []
        if code:
            tab_labels.append("Python")
        if sql:
            tab_labels.append("SQL")
        if model_details:
            tab_labels.append("Model")

        if len(tab_labels) == 1:
            # No need for inner tabs with a single section
            if code:
                _render_code_block(code, language="python", filename="analysis.py")
            elif sql:
                _render_code_block(sql, language="sql", filename="query.sql")
            elif model_details:
                _render_model_details(model_details)
        else:
            tabs = st.tabs(tab_labels)
            idx = 0
            if code:
                with tabs[idx]:
                    _render_code_block(code, language="python", filename="analysis.py")
                idx += 1
            if sql:
                with tabs[idx]:
                    _render_code_block(sql, language="sql", filename="query.sql")
                idx += 1
            if model_details:
                with tabs[idx]:
                    _render_model_details(model_details)


def _render_code_block(content: str, language: str, filename: str):
    st.code(content, language=language)
    st.download_button(
        label=f"Download {filename}",
        data=content,
        file_name=filename,
        mime="text/plain",
        key=f"dl_{filename}_{hash(content) % 100000}",
    )


def _render_model_details(details: dict):
    if details.get("model_type"):
        st.markdown(f"**Model type:** {details['model_type']}")
    if details.get("target"):
        st.markdown(f"**Target variable:** `{details['target']}`")
    if details.get("inputs"):
        inputs = details["inputs"]
        if isinstance(inputs, list):
            inputs = ", ".join(f"`{i}`" for i in inputs)
        st.markdown(f"**Inputs:** {inputs}")
    if details.get("assumptions"):
        st.markdown("**Key assumptions:**")
        assumptions = details["assumptions"]
        if isinstance(assumptions, list):
            for a in assumptions:
                st.markdown(f"- {a}")
        else:
            st.markdown(assumptions)
    if details.get("alternatives"):
        st.markdown("**Alternatives considered:**")
        for alt in details["alternatives"]:
            st.markdown(f"- {alt}")
