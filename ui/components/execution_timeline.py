"""
Execution Timeline Component.

Renders the right-panel trace of every step the system took to produce a result.
Reads from session_state["execution_trace"] which is a list of step dicts.
"""
import streamlit as st
from typing import Optional

_STATUS_ICON = {
    "success": "🟢",
    "failed": "🔴",
    "running": "🟡",
    "pending": "⚪",
    "partial": "🟠",
}

_PLAN_TYPE_LABEL = {
    "template": ("Template", "blue"),
    "learned": ("Learned", "green"),
    "llm_generated": ("LLM-Generated", "orange"),
}


def render_execution_timeline(
    trace: list,
    advanced: bool = False,
    intent: Optional[str] = None,
    plan_type: Optional[str] = None,
    confidence: Optional[float] = None,
):
    """
    Render the execution timeline in the right panel.

    Args:
        trace: List of step dicts with keys:
               step_num, label, tool, status, duration_ms, error, output_preview
        advanced: If True, show raw output per step in expanders.
        intent: Classified intent string (e.g. "visualize").
        plan_type: One of "template", "learned", "llm_generated".
        confidence: Float 0-1 from intent classifier.
    """
    st.markdown("**Execution Trace**")

    if not trace and not intent:
        st.caption("No execution yet. Ask a question to see steps here.")
        return

    # --- Plan metadata header ---
    if intent or plan_type or confidence is not None:
        meta_cols = st.columns([2, 2])
        with meta_cols[0]:
            if intent:
                st.caption(f"Intent: `{intent}`")
            if plan_type:
                label, _ = _PLAN_TYPE_LABEL.get(plan_type, (plan_type, "grey"))
                st.caption(f"Plan: {label}")
        with meta_cols[1]:
            if confidence is not None:
                pct = int(confidence * 100)
                color = "green" if pct >= 80 else "orange" if pct >= 60 else "red"
                st.markdown(
                    f'<span style="color:{color};font-size:0.8em">Confidence: {pct}%</span>',
                    unsafe_allow_html=True,
                )

    if not trace:
        return

    st.divider()

    # --- Step list ---
    for step in trace:
        icon = _STATUS_ICON.get(step.get("status", "pending"), "⚪")
        tool = step.get("tool", "")
        label = step.get("label", tool)
        duration = step.get("duration_ms", 0)
        num = step.get("step_num", "?")
        error = step.get("error")

        duration_str = f"{duration}ms" if duration else ""

        if advanced and (step.get("output_preview") or error):
            with st.expander(f"{icon} [{num}] {label}", expanded=False):
                st.caption(f"Tool: `{tool}`  {duration_str}")
                if error:
                    st.error(error)
                elif step.get("output_preview"):
                    st.code(step["output_preview"], language=None)
        else:
            cols = st.columns([1, 5, 2])
            with cols[0]:
                st.markdown(icon)
            with cols[1]:
                st.markdown(f"**[{num}]** {label}")
                if tool and tool != label:
                    st.caption(f"`{tool}`")
            with cols[2]:
                st.caption(duration_str)

            if error and not advanced:
                st.caption(f"  ↳ {error[:80]}")
