"""
Session Panel Component.

Renders the left sidebar context:
  - Active datasets (name, shape, type badge)
  - Saved workflows (load button)
  - Recent queries (last 5)
  - Session info
"""
import streamlit as st
from typing import List, Optional


def render_session_panel(ctx, saved_workflows: Optional[List[dict]] = None):
    """
    Render the sidebar session/context panel.

    Args:
        ctx: SessionContext instance from ui.session_context.
        saved_workflows: List of saved workflow dicts
                         [{name, timestamp, queries, dataset_names}]
    """
    # --- Active datasets ---
    st.markdown("**Datasets**")
    # Read from session_state directly to work with both legacy and unified keys
    datasets = st.session_state.get("datasets", st.session_state.get("assay_datasets", {}))
    primary_id = st.session_state.get("primary_dataset_id", st.session_state.get("assay_primary_dataset_id"))
    if datasets:
        for name, df in datasets.items():
            is_primary = (name == primary_id)
            badge = " ●" if is_primary else ""
            rows, cols = df.shape
            label = f"**{name}**{badge}" if is_primary else f"{name}{badge}"
            st.markdown(
                f"{label}  \n"
                f"<span style='font-size:0.75em;color:grey'>{rows:,} rows × {cols} cols</span>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No datasets loaded.")

    st.divider()

    # --- Saved workflows ---
    st.markdown("**Saved Workflows**")
    saved_workflows = saved_workflows or st.session_state.get("saved_workflows", [])
    if saved_workflows:
        for wf in saved_workflows[-5:]:  # Show last 5
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"**{wf.get('name', 'Workflow')}**  \n"
                    f"<span style='font-size:0.75em;color:grey'>{wf.get('timestamp', '')[:10]}</span>",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("Load", key=f"load_wf_{wf.get('name', '')}"):
                    st.session_state["pending_workflow"] = wf
                    st.rerun()
    else:
        st.caption("No saved workflows yet.")

    # Save current session as workflow
    if datasets:
        recent = ctx.get_recent_messages(20)
        user_queries = [m["content"] for m in recent if m["role"] == "user"]
        if user_queries:
            with st.expander("Save current workflow", expanded=False):
                wf_name = st.text_input(
                    "Workflow name",
                    value=f"Workflow {len(saved_workflows or []) + 1}",
                    key="wf_name_input",
                )
                if st.button("Save", key="save_workflow_btn"):
                    from datetime import datetime
                    new_wf = {
                        "name": wf_name,
                        "timestamp": datetime.now().isoformat(),
                        "queries": user_queries,
                        "dataset_names": list(datasets.keys()),
                    }
                    existing = st.session_state.get("saved_workflows", [])
                    existing.append(new_wf)
                    st.session_state["saved_workflows"] = existing
                    st.success(f"Saved '{wf_name}'")

    st.divider()

    # --- Recent queries ---
    st.markdown("**Recent Queries**")
    recent = ctx.get_recent_messages(10)
    user_msgs = [m["content"] for m in recent if m["role"] == "user"][-5:]
    if user_msgs:
        for q in reversed(user_msgs):
            short = q[:60] + "…" if len(q) > 60 else q
            if st.button(short, key=f"rerun_{hash(q) % 100000}", use_container_width=True):
                st.session_state["pending_query"] = q
                st.rerun()
    else:
        st.caption("No queries yet.")

    st.divider()

    # --- Advanced mode toggle ---
    advanced = st.toggle(
        "Advanced mode",
        value=st.session_state.get("advanced_mode", False),
        key="advanced_mode_toggle",
        help="Show full execution trace, raw tool outputs, and logs",
    )
    st.session_state["advanced_mode"] = advanced
