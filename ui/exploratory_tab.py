"""
Exploratory Data Analysis Tab with Plotly visualizations and AI-generated insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Any

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from config.feature_flags import ENABLE_LOCAL_LLM


def _get_ai_insight(prompt: str, max_tokens: int = 256) -> str:
    """Get AI-generated insight for the data using LLM Manager or fallback."""
    if not ENABLE_LOCAL_LLM:
        return ""
    
    try:
        # Try new LLM Manager first
        from llm_manager.llm_interface import get_llm_completion, is_llm_available
        if is_llm_available():
            response = get_llm_completion(prompt, max_tokens=max_tokens)
            if response and len(response) > 10:
                return response.strip()
    except ImportError:
        pass
    
    # Fallback disabled - causes C-level crashes
    return ""


def _generate_column_insight(df: pd.DataFrame, col: str) -> str:
    """Generate AI insight for a specific column."""
    col_data = df[col]
    dtype = str(col_data.dtype)
    n_unique = col_data.nunique()
    n_null = col_data.isna().sum()
    
    if dtype in ['int64', 'float64']:
        stats = f"min={col_data.min():.2f}, max={col_data.max():.2f}, mean={col_data.mean():.2f}, std={col_data.std():.2f}"
    else:
        top_vals = col_data.value_counts().head(3).to_dict()
        stats = f"top values: {top_vals}"
    
    prompt = f"""Briefly describe this data column in 1-2 sentences:
Column: {col}
Type: {dtype}
Unique values: {n_unique}
Missing: {n_null}
Stats: {stats}

Description:"""
    
    return _get_ai_insight(prompt, max_tokens=100)


def _generate_dataset_summary(df: pd.DataFrame, context: str = "") -> str:
    """Generate AI summary of the entire dataset."""
    cols_info = []
    for col in df.columns[:10]:  # Limit to first 10 columns
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        cols_info.append(f"- {col} ({dtype}, {n_unique} unique)")
    
    context_str = f"Context: {context}\n" if context else ""
    
    # Get a text sample of the data to help the AI infer meaning
    try:
        sample_str = df.head(3).to_markdown(index=False)
    except ImportError:
        sample_str = df.head(3).to_string(index=False)
    
    prompt = f"""Provide a brief 2-3 sentence summary of this dataset:

{context_str}Shape: {df.shape[0]} rows × {df.shape[1]} columns
Sample Data:
{sample_str}

Columns:
{chr(10).join(cols_info)}

Summary:"""
    
    return _get_ai_insight(prompt, max_tokens=150)


def _generate_correlation_insight(corr_matrix: pd.DataFrame) -> str:
    """Generate insight about correlations."""
    # Find strongest correlations (excluding diagonal)
    corr_pairs = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    corr_pairs.append((col1, col2, corr_val))
    
    if not corr_pairs:
        return "No strong correlations (>0.5) found between numeric columns."
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_corrs = corr_pairs[:5]
    
    prompt = f"""Briefly explain these correlations in 2-3 sentences:
{chr(10).join([f"- {c1} vs {c2}: {v:.2f}" for c1, c2, v in top_corrs])}

Insight:"""
    
    return _get_ai_insight(prompt, max_tokens=150)


def render_exploratory_tab(df: pd.DataFrame, context: str = ""):
    """Render the exploratory data analysis tab."""
    
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not installed. Run: `pip install plotly`")
        st.dataframe(df.describe())
        return
    
    st.header("🔍 Interactive Data Exploration")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing", f"{df.isna().sum().sum():,}")
    with col4:
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory", f"{mem_mb:.1f} MB")
    
    
    # AI Dataset Summary
    with st.expander("🤖 AI Dataset Summary", expanded=True):
        if st.button("Generate Summary", key="gen_summary"):
            with st.spinner("Generating AI summary..."):
                summary = _generate_dataset_summary(df, context=context)
                if summary:
                    st.session_state["ai_summary"] = summary
                else:
                    st.session_state["ai_summary"] = None
        
        # Display persisted summary
        if st.session_state.get("ai_summary"):
            st.info(st.session_state["ai_summary"])
        elif "ai_summary" in st.session_state:
            st.write("AI summary not available. Here's a statistical overview:")
            st.dataframe(df.describe())
        else:
            st.write("Click 'Generate Summary' for an AI-powered dataset description.")
    
    # AI Chart Suggestions
    with st.expander("🎨 AI Chart Suggestions", expanded=False):
        # Mode toggle
        suggestion_mode = st.radio(
            "Suggestion Mode", 
            ["Smart (Fast)", "LLM (AI-Powered)"],
            horizontal=True,
            help="Smart uses rule-based analysis; LLM uses AI but is slower"
        )
        
        if st.button("Suggest Visualizations", key="suggest_charts"):
            with st.spinner("Analyzing your data..."):
                if suggestion_mode == "Smart (Fast)":
                    # Use rule-based smart chart selection
                    try:
                        from visualization.smart_charts import recommend_charts, render_recommendation
                        recommendations = recommend_charts(df, max_recommendations=5)
                        st.session_state["chart_suggestions"] = {
                            "model": "Smart Charts (Rule-based)",
                            "suggestions": [
                                {
                                    "type": rec.chart_type.value,
                                    "x": rec.x_col,
                                    "y": rec.y_col,
                                    "title": rec.title,
                                    "reason": rec.reason,
                                    "confidence": rec.confidence,
                                    "color": rec.color_col,
                                    "_recommendation": rec  # Keep original for rendering
                                }
                                for rec in recommendations
                            ]
                        }
                    except Exception as e:
                        st.session_state["chart_suggestions"] = {"error": f"Smart charts error: {e}"}
                else:
                    # Use LLM-based suggestions
                    try:
                        from llm_manager.llm_interface import suggest_visualizations, is_llm_available, get_active_model_name
                        
                        if not is_llm_available():
                            st.session_state["chart_suggestions"] = {"error": "no_llm"}
                        else:
                            model_name = get_active_model_name()
                            suggestions = suggest_visualizations(df)
                            st.session_state["chart_suggestions"] = {
                                "model": model_name,
                                "suggestions": suggestions
                            }
                    except ImportError as e:
                        st.session_state["chart_suggestions"] = {"error": f"import: {e}"}
                    except Exception as e:
                        st.session_state["chart_suggestions"] = {"error": f"exception: {e}"}
        
        # Display persisted suggestions
        chart_data = st.session_state.get("chart_suggestions", {})
        
        if "error" in chart_data:
            error = chart_data["error"]
            if error == "no_llm":
                st.warning("⚠️ No LLM available. Configure one in the '🤖 LLM Settings' tab first.")
            else:
                st.error(f"Error: {error}")
        elif "suggestions" in chart_data:
            suggestions = chart_data["suggestions"]
            model_name = chart_data.get("model", "Unknown")
            st.caption(f"Using model: {model_name}")
            
            if suggestions:
                st.success(f"Found {len(suggestions)} chart recommendations!")
                for i, sug in enumerate(suggestions):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{i+1}. {sug.get('title', 'Chart')}**")
                        st.caption(f"{sug.get('type', 'chart').title()} chart: {sug.get('x', '')} vs {sug.get('y', 'count')}")
                        st.write(f"💡 {sug.get('reason', '')}")
                    with col2:
                        if st.button("View", key=f"view_sug_{i}"):
                            st.session_state[f"show_sug_{i}"] = True
                    
                    # Show chart if requested
                    if st.session_state.get(f"show_sug_{i}"):
                        chart_type = sug.get('type', 'bar').lower()
                        x = sug.get('x')
                        y = sug.get('y')
                        
                        if x in df.columns:
                            # Smart chart type detection for time series
                            x_col = df[x]
                            is_time_series = False
                            
                            # Check if x looks like time data
                            if 'date' in x.lower() or 'time' in x.lower() or 'year' in x.lower():
                                is_time_series = True
                            elif x_col.dtype in ['datetime64[ns]', 'datetime64']:
                                is_time_series = True
                            elif len(df) > 50 and x_col.dtype in ['int64', 'float64']:
                                # Many numeric points in sequence - likely time series
                                if x_col.is_monotonic_increasing or x_col.is_monotonic_decreasing:
                                    is_time_series = True
                            
                            # Override bar chart to line for time series with many points
                            if chart_type == 'bar' and is_time_series and len(df) > 30:
                                chart_type = 'line'  # Much more readable for time series
                            
                            if chart_type == 'bar':
                                # Limit bar charts to top N if too many categories
                                if len(df) > 50:
                                    df_plot = df.nlargest(30, y) if y in df.columns else df.head(30)
                                else:
                                    df_plot = df
                                fig = px.bar(df_plot, x=x, y=y if y in df.columns else None, title=sug.get('title', ''))
                            elif chart_type == 'line':
                                fig = px.line(df.sort_values(x) if is_time_series else df, x=x, y=y if y in df.columns else None, title=sug.get('title', ''))
                            elif chart_type == 'scatter':
                                fig = px.scatter(df, x=x, y=y if y in df.columns else None, title=sug.get('title', ''))
                            elif chart_type == 'histogram':
                                fig = px.histogram(df, x=x, title=sug.get('title', ''))
                            elif chart_type == 'pie':
                                fig = px.pie(df, names=x, title=sug.get('title', ''))
                            else:
                                fig = px.line(df, x=x, y=y if y in df.columns else None, title=sug.get('title', ''))
                            
                            fig.update_layout(template="plotly_dark")
                            st.plotly_chart(fig, width="stretch", key=f"sug_chart_{i}")
            else:
                st.info("💡 LLM couldn't parse chart suggestions. This usually means the model needs to be loaded first.\n\n**Try:** Go to '🤖 LLM Settings' tab and click 'Load Model', then try again.")
        else:
            st.write("Let AI recommend the best visualizations for your data.")
    
    # Column selector
    st.subheader("📊 Column Analysis")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Distribution plots
    if numeric_cols:
        st.markdown("### Distribution Analysis")
        
        dist_col = st.selectbox("Select numeric column", numeric_cols, key="dist_col")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                df, x=dist_col, 
                title=f"Distribution of {dist_col}",
                template="plotly_dark",
                color_discrete_sequence=["#00d4ff"]
            )
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_hist, width="stretch", key=f"hist_{dist_col}")
        
        with col2:
            # Box plot
            fig_box = px.box(
                df, y=dist_col,
                title=f"Box Plot of {dist_col}",
                template="plotly_dark",
                color_discrete_sequence=["#ff6b6b"]
            )
            fig_box.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_box, width="stretch", key=f"box_{dist_col}")
        
        # AI insight for selected column
        if st.button(f"🤖 Get AI Insight for {dist_col}", key="col_insight"):
            with st.spinner("Generating insight..."):
                insight = _generate_column_insight(df, dist_col)
                if insight:
                    st.info(insight)
    
    # Categorical analysis
    if categorical_cols:
        st.markdown("### Categorical Analysis")
        
        cat_col = st.selectbox("Select categorical column", categorical_cols, key="cat_col")
        
        value_counts = df[cat_col].value_counts().head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = px.bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                title=f"Value Counts: {cat_col}",
                template="plotly_dark",
                color=value_counts.values,
                color_continuous_scale="viridis"
            )
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title=cat_col,
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig_bar, width="stretch", key=f"bar_{cat_col}")
        
        with col2:
            fig_pie = px.pie(
                values=value_counts.values,
                names=value_counts.index.astype(str),
                title=f"Proportion: {cat_col}",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_pie, width="stretch", key=f"pie_{cat_col}")
    
    # Correlation heatmap
    if len(numeric_cols) >= 2:
        st.markdown("### Correlation Analysis")
        
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            title="Correlation Heatmap",
            template="plotly_dark",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        fig_corr.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_corr, width="stretch", key="corr_heatmap")
        
        # AI correlation insight
        if st.button("🤖 Explain Correlations", key="corr_insight"):
            with st.spinner("Analyzing correlations..."):
                insight = _generate_correlation_insight(corr_matrix)
                st.info(insight)
    
    # Scatter plot explorer
    if len(numeric_cols) >= 2:
        st.markdown("### Scatter Plot Explorer")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        with col2:
            y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="scatter_y")
        with col3:
            color_col = st.selectbox("Color by", ["None"] + categorical_cols + numeric_cols, key="scatter_color")
        
        color_param = None if color_col == "None" else color_col
        
        fig_scatter = px.scatter(
            df.sample(min(5000, len(df))),  # Sample for performance
            x=x_col,
            y=y_col,
            color=color_param,
            title=f"{x_col} vs {y_col}",
            template="plotly_dark",
            opacity=0.6
        )
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_scatter, width="stretch", key=f"scatter_{x_col}_{y_col}")
    
    # Time series if datetime column exists
    if datetime_cols and numeric_cols:
        st.markdown("### Time Series Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            time_col = st.selectbox("Time column", datetime_cols, key="time_col")
        with col2:
            value_col = st.selectbox("Value column", numeric_cols, key="time_value")
        
        df_sorted = df.sort_values(time_col)
        
        fig_time = px.line(
            df_sorted,
            x=time_col,
            y=value_col,
            title=f"{value_col} Over Time",
            template="plotly_dark"
        )
        fig_time.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_time, width="stretch", key=f"timeseries_{time_col}_{value_col}")
    
    # Data quality summary
    st.markdown("### Data Quality")
    
    quality_data = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct_missing = (n_missing / len(df)) * 100
        n_unique = df[col].nunique()
        dtype = str(df[col].dtype)
        
        quality_data.append({
            "Column": col,
            "Type": dtype,
            "Missing": n_missing,
            "Missing %": f"{pct_missing:.1f}%",
            "Unique": n_unique,
            "Unique %": f"{(n_unique / len(df)) * 100:.1f}%"
        })
    
    quality_df = pd.DataFrame(quality_data)
    st.dataframe(quality_df, width="stretch")
