"""
Result Card Component.

Every significant output is wrapped in a structured header that makes provenance
explicit: what dataset, what model, what features, how confident, key insight.
"""
import streamlit as st
from typing import Optional, List
from ui.components.code_inspector import render_code_inspector


_CONFIDENCE_COLOR = {
    "high": "green",
    "medium": "orange",
    "low": "red",
}


def render_result_card(
    title: str,
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    features: Optional[List[str]] = None,
    confidence: Optional[str] = None,      # "high" | "medium" | "low"
    confidence_note: Optional[str] = None,  # e.g. "limited historical seasonality"
    insight: Optional[str] = None,
    code: Optional[str] = None,
    sql: Optional[str] = None,
    model_details: Optional[dict] = None,
):
    """
    Render a structured result header above any output.

    Usage:
        render_result_card(
            title="Revenue Forecast (Next 30 Days)",
            dataset="sales_data.csv",
            model="Prophet Time Series",
            features=["Date", "Revenue"],
            confidence="medium",
            confidence_note="limited historical seasonality",
            insight="Revenue expected to increase ~12% MoM",
            code=generated_python_code,
        )
        # Then render the actual chart/table/etc. below
    """
    with st.container(border=True):
        # Title row
        st.markdown(f"##### {title}")

        # Provenance row
        if dataset or model or features:
            parts = []
            if dataset:
                parts.append(f"Dataset: **{dataset}**")
            if model:
                parts.append(f"Model: **{model}**")
            if features:
                parts.append(f"Features: {', '.join(features)}")
            st.caption("Generated using: " + "  ·  ".join(parts))

        # Confidence + insight row
        info_cols = st.columns([1, 2])
        with info_cols[0]:
            if confidence:
                color = _CONFIDENCE_COLOR.get(confidence.lower(), "grey")
                note = f" — {confidence_note}" if confidence_note else ""
                st.markdown(
                    f'<span style="color:{color};font-size:0.85em">'
                    f"Confidence: {confidence.capitalize()}{note}</span>",
                    unsafe_allow_html=True,
                )
        with info_cols[1]:
            if insight:
                st.info(f"**Key Insight:** {insight}", icon="💡")

        # Show your work — inline inspector
        if code or sql or model_details:
            render_code_inspector(code=code, sql=sql, model_details=model_details)
