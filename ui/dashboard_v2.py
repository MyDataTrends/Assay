# === Ensure project root is in Python path ===
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
os.environ.setdefault("STREAMLIT_SERVER_ENABLECORS", "true")

import streamlit as st
import pandas as pd
import json
import hashlib

# Configure page - MUST be first Streamlit command
st.set_page_config(
    page_title="Assay Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Imports after page config
from modeling.llm_dynamic_analyzer import dynamic_analyze as run_workflow
from orchestration.data_quality_scorer import compute_safety_metrics, summarize_for_display
from orchestration.analysis_selector import select_analyzer
from preprocessing.metadata_parser import infer_column_meta, merge_user_labels
from preprocessing.sanitize import scrub_df
from storage.local_backend import load_datalake_dfs
from feedback.ratings import store_rating
from ui import redaction_banner
from ui.exploratory_tab import render_exploratory_tab
from datetime import datetime

# ============================================================================
# AUTO-SAVE RESULTS
# ============================================================================
OUTPUT_DIR = _PROJECT_ROOT / "output_files"
OUTPUT_DIR.mkdir(exist_ok=True)

def _auto_save_results(result: dict, data_name: str) -> Path:
    """Auto-save analysis results to output_files/ with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in data_name)
    filename = f"{safe_name}_{timestamp}.json"
    filepath = OUTPUT_DIR / filename
    
    # Save the result
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    
    # Also save a "latest" symlink/copy for easy access
    latest_path = OUTPUT_DIR / f"{safe_name}_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    
    # Update index file
    index_path = OUTPUT_DIR / "index.json"
    try:
        with open(index_path, "r") as f:
            index = json.load(f)
    except:
        index = {"runs": []}
    
    index["runs"].append({
        "filename": filename,
        "data_name": data_name,
        "timestamp": timestamp,
        "analysis_type": result.get("analysis_type", "unknown"),
    })
    index["runs"] = index["runs"][-50:]  # Keep last 50
    
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    
    return filepath

# ============================================================================
# CUSTOM CSS FOR BETTER UI
# ============================================================================
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Welcome hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        border-color: #667eea;
    }
    
    /* Step indicators */
    .step-indicator {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2rem;
        height: 2rem;
        background: #667eea;
        border-radius: 50%;
        color: white;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    /* Chat container */
    .chat-container {
        border: 1px solid #333;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "data" not in st.session_state:
    st.session_state.data = None
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Assay", width=150)
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    
    if st.button("🏠 Home", type="secondary"):
        st.session_state.page = "home"
        st.rerun()
    
    if st.button("📊 Analyze Data", 
                 type="primary" if st.session_state.data is not None else "secondary",
                 disabled=st.session_state.data is None):
        st.session_state.page = "analyze"
        st.rerun()
    
    if st.button("📈 Results",
                 type="primary" if st.session_state.result is not None else "secondary",
                 disabled=st.session_state.result is None):
        st.session_state.page = "results"
        st.rerun()
    
    if st.button("💬 Chat"):
        st.session_state.page = "chat"
        st.rerun()
    
    if st.button("⚙️ Settings"):
        st.session_state.page = "settings"
        st.rerun()
    
    st.markdown("---")
    
    # Quick upload in sidebar
    st.markdown("### 📁 Quick Upload")
    uploaded_file = st.file_uploader(
        "Drop your CSV here",
        type=["csv"],
        label_visibility="collapsed",
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.session_state.data_name = uploaded_file.name
            st.success(f"✅ Loaded {len(df):,} rows")
            if st.session_state.page == "home":
                st.session_state.page = "analyze"
                st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    # Data status
    if st.session_state.data is not None:
        st.markdown("---")
        st.markdown("### 📋 Current Data")
        st.caption(f"**{st.session_state.get('data_name', 'Dataset')}**")
        st.caption(f"{len(st.session_state.data):,} rows × {len(st.session_state.data.columns)} cols")
        
        if st.button("🗑️ Clear Data"):
            st.session_state.data = None
            st.session_state.result = None
            st.session_state.page = "home"
            st.rerun()

# ============================================================================
# HOME PAGE
# ============================================================================
def render_home():
    # Hero section
    st.markdown("""
    <div class="hero-section">
        <h1>🔮 Welcome to Assay</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">
            Your AI-powered data analysis assistant. Upload data, get insights, make decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start steps
    st.markdown("## 🚀 Get Started in 3 Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3><span class="step-indicator">1</span> Upload Data</h3>
            <p>Drag and drop your CSV file or use the sidebar uploader.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3><span class="step-indicator">2</span> Auto-Analyze</h3>
            <p>Assay automatically detects patterns, anomalies, and insights.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3><span class="step-indicator">3</span> Get Insights</h3>
            <p>View results, chat with your data, and export findings.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main upload area
    st.markdown("## 📁 Upload Your Data")
    
    upload_col, sample_col = st.columns([2, 1])
    
    with upload_col:
        uploaded = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file to begin analysis",
            key="main_uploader"
        )
        
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.session_state.data = df
                st.session_state.data_name = uploaded.name
                st.success(f"✅ Successfully loaded **{uploaded.name}** ({len(df):,} rows × {len(df.columns)} columns)")
                
                if st.button("🚀 Start Analysis", type="primary"):
                    st.session_state.page = "analyze"
                    st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with sample_col:
        st.markdown("### 📦 Or Try Sample Data")
        
        sample_datasets = {
            "Telco Churn": "datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv",
            "Store Sales": "datasets/transactions.csv",
            "Oil Prices": "datasets/oil.csv",
        }
        
        for name, path in sample_datasets.items():
            if st.button(f"Load {name}", key=f"sample_{name}"):
                try:
                    df = pd.read_csv(_PROJECT_ROOT / path)
                    st.session_state.data = df
                    st.session_state.data_name = name
                    st.session_state.page = "analyze"
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load sample: {e}")
    
    st.markdown("---")
    
    # Features overview
    st.markdown("## ✨ What Assay Can Do")
    
    feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
    
    with feat_col1:
        st.markdown("### 🔍 Auto-Detection")
        st.caption("Automatically identifies column types, patterns, and relationships")
    
    with feat_col2:
        st.markdown("### 🤖 Smart Modeling")
        st.caption("Selects the best ML model for your data automatically")
    
    with feat_col3:
        st.markdown("### 🔗 Data Enrichment")
        st.caption("Joins with public datasets to enhance your analysis")
    
    with feat_col4:
        st.markdown("### 💬 Chat Interface")
        st.caption("Ask questions about your data in natural language")

# ============================================================================
# ANALYZE PAGE
# ============================================================================
def render_analyze():
    st.markdown("# 📊 Data Analysis")
    
    if st.session_state.data is None:
        st.warning("⚠️ No data loaded. Please upload a file first.")
        if st.button("← Go to Home"):
            st.session_state.page = "home"
            st.rerun()
        return
    
    data = st.session_state.data
    
    # Data overview tabs
    tab1, tab2, tab_explore, tab3, tab4 = st.tabs(["📋 Preview", "📊 Statistics", "🔍 Explore", "🎯 Configure", "🚀 Run Analysis"])
    
    with tab1:
        st.markdown("### Data Preview")
        st.dataframe(data.head(100), height=400)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{len(data):,}")
        with col2:
            st.metric("Columns", len(data.columns))
        with col3:
            st.metric("Missing Values", f"{data.isna().sum().sum():,}")
        with col4:
            memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory", f"{memory_mb:.1f} MB")
    
    with tab2:
        st.markdown("### Column Statistics")
        
        # Data quality score
        metrics = compute_safety_metrics(data)
        summary = summarize_for_display(metrics)
        
        qual_col1, qual_col2, qual_col3 = st.columns(3)
        with qual_col1:
            st.metric("Quality Score", f"{summary['score']}%", 
                     delta="Good" if summary['score'] >= 80 else "Needs Review")
        with qual_col2:
            st.metric("Completeness", f"{summary['completeness_pct']}%")
        with qual_col3:
            st.metric("Complete Rows", f"{summary['rows_complete_pct']}%")
        
        if summary['warnings']:
            st.warning("**Data Quality Warnings:**")
            for w in summary['warnings']:
                st.write(f"- {w}")
        
        st.markdown("---")
        st.markdown("### Column Details")
        
        # Infer column metadata
        meta = infer_column_meta(data)
        
        col_data = []
        for m in meta:
            col = data[m.name]
            col_data.append({
                "Column": m.name,
                "Type": str(col.dtype),
                "Role": m.role,
                "Unique": col.nunique(),
                "Missing": col.isna().sum(),
                "Missing %": f"{col.isna().mean()*100:.1f}%",
            })
        
        st.dataframe(pd.DataFrame(col_data), hide_index=True)
    
    with tab_explore:
        # Interactive Plotly visualizations with AI insights
        render_exploratory_tab(data, meta)
    
    with tab3:
        st.markdown("### Analysis Configuration")
        
        # Target column selection
        st.markdown("#### 🎯 Target Column")
        target_options = ["Auto-detect"] + list(data.columns)
        target = st.selectbox(
            "Select the column you want to predict/analyze",
            target_options,
            help="Leave as 'Auto-detect' to let Assay choose the best target"
        )
        st.session_state.target_column = None if target == "Auto-detect" else target
        
        # Analysis type
        st.markdown("#### 📈 Analysis Type")
        analysis_type = st.radio(
            "Choose analysis approach",
            ["🤖 Auto (Recommended)", "📊 Descriptive Only", "🔮 Predictive Modeling"],
            horizontal=True,
        )
        st.session_state.analysis_type = analysis_type
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Enable SHAP explanations", value=True, key="enable_shap")
                st.checkbox("Auto-enrich with public data", value=True, key="enable_enrichment")
            with col2:
                st.checkbox("Run diagnostics", value=True, key="enable_diagnostics")
                st.checkbox("Generate business summary", value=True, key="enable_summary")
    
    with tab4:
        st.markdown("### 🚀 Run Analysis")
        
        st.info("""
        **Ready to analyze!** Click the button below to start. Assay will:
        1. Clean and preprocess your data
        2. Detect patterns and anomalies
        3. Select and train the best model
        4. Generate insights and explanations
        """)
        
        if st.button("🚀 Start Analysis", type="primary"):
            with st.spinner("Analyzing your data... This may take a moment."):
                try:
                    # Run the workflow
                    result = run_workflow(
                        data,
                        target=st.session_state.get("target_column"),
                    )
                    st.session_state.result = result
                    
                    # Auto-save results to output_files/
                    _auto_save_results(result, st.session_state.get("data_name", "analysis"))
                    
                    st.success("✅ Analysis complete!")
                    st.session_state.page = "results"
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

# ============================================================================
# RESULTS PAGE
# ============================================================================
def render_results():
    if st.session_state.result is None:
        st.warning("⚠️ No results available. Please run an analysis first.")
        if st.button("← Go to Analyze"):
            st.session_state.page = "analyze"
            st.rerun()
        return
    
    result = st.session_state.result
    data = st.session_state.data
    
    # Header with export button
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown("# 📈 Analysis Report")
    with header_col2:
        st.download_button(
            "📥 Export JSON",
            json.dumps(result, indent=2, default=str),
            file_name="analysis_results.json",
            mime="application/json",
        )
    
    # ========== EXECUTIVE SUMMARY ==========
    st.markdown("---")
    st.markdown("## 📋 Executive Summary")
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <h3 style="margin: 0; color: white;">📊 Dataset</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: white; font-weight: bold;">
                {:,} rows
            </p>
            <p style="margin: 0; color: rgba(255,255,255,0.8);">{} columns</p>
        </div>
        """.format(len(data), len(data.columns)), unsafe_allow_html=True)
    
    with col2:
        analysis_type = result.get("analysis_type", "descriptive")
        type_emoji = {"regression": "📈", "classification": "🏷️", "clustering": "🎯", 
                      "forecasting": "🔮", "descriptive": "📊"}.get(analysis_type, "📊")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <h3 style="margin: 0; color: white;">{type_emoji} Analysis</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: white; font-weight: bold;">
                {analysis_type.title() if analysis_type else "Descriptive"}
            </p>
            <p style="margin: 0; color: rgba(255,255,255,0.8);">Auto-selected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        metrics = result.get("metrics", {})
        score = metrics.get("score") or metrics.get("accuracy") or metrics.get("r2")
        score_display = f"{score:.1%}" if score and score <= 1 else (f"{score:.2f}" if score else "N/A")
        score_color = "#38ef7d" if score and score > 0.7 else ("#ffd93d" if score and score > 0.5 else "#ff6b6b")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <h3 style="margin: 0; color: white;">🎯 Score</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: white; font-weight: bold;">
                {score_display}
            </p>
            <p style="margin: 0; color: rgba(255,255,255,0.8);">Model performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        status = "✅ Complete" if not result.get("needs_role_review") else "⚠️ Review"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <h3 style="margin: 0; color: white;">📌 Status</h3>
            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: white; font-weight: bold;">
                {status}
            </p>
            <p style="margin: 0; color: rgba(255,255,255,0.8);">Analysis status</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========== DATA VISUALIZATIONS ==========
    st.markdown("---")
    st.markdown("## 📊 Data Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Always show data distribution
        st.markdown("### 📈 Data Distribution")
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select column", numeric_cols, key="dist_col")
            st.bar_chart(data[selected_col].value_counts().head(20))
        else:
            st.info("No numeric columns to visualize")
    
    with viz_col2:
        # Show time series if date column exists
        date_cols = [c for c in data.columns if 'date' in c.lower() or data[c].dtype == 'datetime64[ns]']
        if date_cols and numeric_cols:
            st.markdown("### 📅 Trend Over Time")
            date_col = date_cols[0]
            value_col = st.selectbox("Select value", numeric_cols, key="trend_col")
            try:
                trend_data = data.copy()
                trend_data[date_col] = pd.to_datetime(trend_data[date_col])
                trend_data = trend_data.sort_values(date_col)
                st.line_chart(trend_data.set_index(date_col)[value_col])
            except:
                st.info("Could not create time series chart")
        else:
            # Show correlation or top values
            st.markdown("### 🔝 Top Values")
            if numeric_cols:
                top_col = numeric_cols[0]
                st.dataframe(data.nlargest(10, top_col)[[top_col]], hide_index=True)
    
    # Feature importances if available
    explanations = result.get("model_info", {}).get("explanations", {})
    fi = explanations.get("feature_importances") or explanations.get("coefficients") or explanations.get("shap_importance")
    if fi is not None:
        st.markdown("### 🎯 Feature Importance")
        try:
            if isinstance(fi, list) and len(fi) > 0:
                # Create feature importance chart
                feature_names = list(data.columns)[:len(fi)]
                fi_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': fi[:len(feature_names)]
                }).sort_values('Importance', ascending=True).tail(10)
                st.bar_chart(fi_df.set_index('Feature'))
        except:
            st.info("Could not display feature importances")
    
    # ========== KEY FINDINGS ==========
    st.markdown("---")
    st.markdown("## 💡 Key Findings")
    
    findings = []
    
    # Generate findings from the data and results
    # 1. Data shape
    findings.append(f"📊 **Dataset Size**: Your data contains **{len(data):,} records** across **{len(data.columns)} variables**.")
    
    # 2. Missing data
    missing_pct = (data.isna().sum().sum() / data.size) * 100
    if missing_pct > 0:
        findings.append(f"⚠️ **Data Quality**: Found **{missing_pct:.1f}% missing values**. Consider data cleaning for better results.")
    else:
        findings.append("✅ **Data Quality**: No missing values detected. Your data is complete!")
    
    # 3. Analysis type reasoning
    decision = result.get("modeling_decision", {})
    if decision.get("reasoning"):
        findings.append(f"🤖 **Analysis Choice**: {decision['reasoning']}")
    
    # 4. Model performance
    if score:
        if score > 0.8:
            findings.append(f"🌟 **Strong Performance**: Model achieved **{score_display}** accuracy - excellent predictive power!")
        elif score > 0.6:
            findings.append(f"👍 **Good Performance**: Model achieved **{score_display}** accuracy - solid results.")
        else:
            findings.append(f"📈 **Room for Improvement**: Model achieved **{score_display}** - consider adding more features or data.")
    
    # 5. Column insights
    cat_cols = data.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        findings.append(f"🏷️ **Categorical Variables**: Found **{len(cat_cols)} text/category columns** that could be used for grouping.")
    
    # 6. Numeric insights
    if numeric_cols:
        findings.append(f"📈 **Numeric Variables**: Found **{len(numeric_cols)} numeric columns** suitable for statistical analysis.")
    
    # Display findings as a nice report
    for finding in findings:
        st.markdown(f"""
        <div style="background: #1e1e2e; padding: 1rem; border-radius: 0.5rem; 
                    margin: 0.5rem 0; border-left: 4px solid #667eea;">
            {finding}
        </div>
        """, unsafe_allow_html=True)
    
    # ========== DETAILED METRICS ==========
    if metrics:
        st.markdown("---")
        st.markdown("## 📏 Detailed Metrics")
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.markdown("### Performance Metrics")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    st.markdown(f"- **{key.replace('_', ' ').title()}**: {value:.4f}" if isinstance(value, float) else f"- **{key.replace('_', ' ').title()}**: {value}")
        
        with metrics_col2:
            if result.get("model_info", {}).get("merge_report"):
                st.markdown("### Data Enrichment")
                merge = result["model_info"]["merge_report"]
                st.markdown(f"- **Tables Considered**: {merge.get('tables_considered', 'N/A')}")
                st.markdown(f"- **Successful Merges**: {merge.get('successful_merges', 0)}")
                st.markdown(f"- **New Columns Added**: {merge.get('total_new_columns', 0)}")
    
    # ========== RECOMMENDATIONS ==========
    st.markdown("---")
    st.markdown("## 🎯 Recommendations")
    
    recommendations = []
    
    if result.get("needs_role_review"):
        recommendations.append("🔍 **Review Column Roles**: The model suggests reviewing how columns are interpreted. Go to Analyze → Configure to adjust.")
    
    if missing_pct > 10:
        recommendations.append("🧹 **Clean Your Data**: High missing value percentage. Consider imputation or removing incomplete records.")
    
    if not score or score < 0.6:
        recommendations.append("📊 **Add More Features**: Model performance could improve with additional relevant data columns.")
    
    if len(data) < 1000:
        recommendations.append("📈 **More Data Needed**: Small dataset detected. More records typically improve model accuracy.")
    
    if not recommendations:
        recommendations.append("✅ **Looking Good!**: Your analysis completed successfully. Explore the visualizations above for insights.")
    
    for rec in recommendations:
        st.info(rec)

# ============================================================================
# CHAT PAGE
# ============================================================================
def render_chat():
    st.markdown("# 💬 Chat with Your Data")
    
    if st.session_state.data is None:
        st.warning("⚠️ No data loaded. Please upload a file first to chat about it.")
        if st.button("← Go to Home"):
            st.session_state.page = "home"
            st.rerun()
        return
    
    # Chat interface
    st.markdown("Ask questions about your data in natural language.")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assay:** {msg['content']}")
    
    # Chat input
    st.markdown("---")
    
    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("### 💡 Suggested Questions")
        suggestions = [
            "What are the main patterns in this data?",
            "Which columns have the most missing values?",
            "Show me a summary of the numeric columns",
            "What would be a good target variable to predict?",
        ]
        
        cols = st.columns(2)
        for i, sugg in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(sugg, key=f"sugg_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": sugg})
                    # Generate response (simplified for now)
                    response = _generate_chat_response(sugg, st.session_state.data)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
    
    # Text input
    user_input = st.chat_input("Ask a question about your data...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = _generate_chat_response(user_input, st.session_state.data)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


def _generate_chat_response(question: str, data: pd.DataFrame) -> str:
    """Generate a response to a user question about the data."""
    question_lower = question.lower()
    
    # Simple pattern matching for common questions
    if any(kw in question_lower for kw in ["missing", "null", "na ", "nan"]):
        missing = data.isna().sum()
        missing = missing[missing > 0]
        if len(missing) == 0:
            return "Great news! Your data has no missing values."
        return f"Here are the columns with missing values:\n\n" + "\n".join(
            f"- **{col}**: {count:,} missing ({count/len(data)*100:.1f}%)"
            for col, count in missing.items()
        )
    
    if any(kw in question_lower for kw in ["summary", "describe", "statistics", "stats"]):
        desc = data.describe().round(2)
        return f"Here's a statistical summary of your numeric columns:\n\n{desc.to_markdown()}"
    
    if any(kw in question_lower for kw in ["column", "feature", "field", "variable"]):
        lines = [f"Your dataset has **{len(data.columns)} columns**:\n"]
        for col in data.columns:
            dtype = str(data[col].dtype)
            nunique = data[col].nunique()
            missing_pct = data[col].isna().mean() * 100
            lines.append(f"- **{col}** — {dtype}, {nunique:,} unique values" + 
                        (f", {missing_pct:.1f}% missing" if missing_pct > 0 else ""))
        return "\n".join(lines)
    
    if any(kw in question_lower for kw in ["rows", "size", "shape", "how many", "count"]):
        mem_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        return f"Your dataset has **{len(data):,} rows** and **{len(data.columns)} columns** ({mem_mb:.1f} MB in memory)."
    
    if any(kw in question_lower for kw in ["time series", "timeseries", "temporal", "date", "time"]):
        # Look for date/time columns
        date_cols = []
        for col in data.columns:
            if data[col].dtype == "datetime64[ns]":
                date_cols.append((col, "datetime"))
            elif "date" in col.lower() or "time" in col.lower():
                date_cols.append((col, str(data[col].dtype)))
        
        if date_cols:
            lines = ["📅 **Time-related columns found:**\n"]
            for col, dtype in date_cols:
                sample = data[col].dropna().iloc[0] if len(data[col].dropna()) > 0 else "N/A"
                lines.append(f"- **{col}** ({dtype}) — e.g., `{sample}`")
            lines.append("\nThis data appears suitable for time-series analysis!")
            return "\n".join(lines)
        return "No obvious date/time columns found. Check if dates are stored as strings or integers."
    
    if any(kw in question_lower for kw in ["pattern", "insight", "interesting", "notice"]):
        insights = []
        # Check for date columns
        date_cols = [c for c in data.columns if "date" in c.lower() or data[c].dtype == "datetime64[ns]"]
        if date_cols:
            insights.append(f"📅 Found date columns: {', '.join(date_cols)} — this could be time-series data")
        
        # Check for high cardinality (ID columns)
        for col in data.columns:
            if data[col].nunique() == len(data):
                insights.append(f"🔑 Column '{col}' has all unique values — likely an ID column")
        
        # Check for binary columns
        binary_cols = [c for c in data.columns if data[c].nunique() == 2]
        if binary_cols:
            insights.append(f"✅ Binary columns found: {', '.join(binary_cols)} — good for classification")
        
        # Check for low cardinality (categorical)
        cat_cols = [c for c in data.columns if 2 < data[c].nunique() <= 20]
        if cat_cols:
            insights.append(f"📊 Categorical columns: {', '.join(cat_cols[:5])} — good for grouping/segmentation")
        
        # Check for numeric columns
        num_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if num_cols:
            insights.append(f"📈 Numeric columns: {', '.join(num_cols[:5])} — suitable for regression/correlation")
        
        if insights:
            return "Here are some patterns I noticed:\n\n" + "\n".join(insights)
        return "I'd need to run a deeper analysis to find patterns. Try running the full analysis!"
    
    if any(kw in question_lower for kw in ["target", "predict", "forecast", "model"]):
        # Suggest potential targets
        suggestions = []
        for col in data.columns:
            if data[col].nunique() <= 10 and data[col].nunique() > 1:
                suggestions.append(f"- **{col}** (categorical, {data[col].nunique()} classes) — classification")
            elif data[col].dtype in ["int64", "float64"] and "id" not in col.lower():
                suggestions.append(f"- **{col}** (numeric) — regression")
        
        if suggestions:
            return "Here are potential target columns for prediction:\n\n" + "\n".join(suggestions[:8])
        return "I couldn't identify obvious target columns. What are you trying to predict?"
    
    if any(kw in question_lower for kw in ["correlation", "correlate", "related", "relationship"]):
        num_data = data.select_dtypes(include=["int64", "float64"])
        if len(num_data.columns) < 2:
            return "Need at least 2 numeric columns to compute correlations."
        corr = num_data.corr()
        # Find top correlations
        pairs = []
        for i, col1 in enumerate(corr.columns):
            for col2 in corr.columns[i+1:]:
                pairs.append((col1, col2, corr.loc[col1, col2]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        lines = ["**Top correlations:**\n"]
        for col1, col2, r in pairs[:5]:
            strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
            direction = "positive" if r > 0 else "negative"
            lines.append(f"- **{col1}** ↔ **{col2}**: {r:.3f} ({strength} {direction})")
        return "\n".join(lines)
    
    if any(kw in question_lower for kw in ["type", "dtype", "data type"]):
        type_counts = data.dtypes.value_counts()
        lines = ["**Column data types:**\n"]
        for dtype, count in type_counts.items():
            cols = data.select_dtypes(include=[dtype]).columns.tolist()
            lines.append(f"- **{dtype}**: {count} columns — {', '.join(cols[:5])}" + 
                        (f"... (+{len(cols)-5} more)" if len(cols) > 5 else ""))
        return "\n".join(lines)
    
    # Try LLM if available
    try:
        from preprocessing.llm_preprocessor import llm_completion
        # Build context about the data
        context = f"""Dataset info:
- Rows: {len(data):,}
- Columns: {', '.join(data.columns[:20])}
- Dtypes: {dict(data.dtypes.value_counts())}
- Sample values: {data.head(3).to_dict()}

User question: {question}

Provide a helpful, concise answer about this data."""
        
        response = llm_completion(context)
        if response and len(response) > 10:
            return response
    except Exception:
        pass
    
    # Default response
    return (
        "I'm not sure how to answer that specific question. "
        "Try asking about:\n"
        "- **Column information** — list all columns with types\n"
        "- **Missing values** — find columns with nulls\n"
        "- **Data summary** — statistical overview\n"
        "- **Time series** — find date/time columns\n"
        "- **Correlations** — relationships between numeric columns\n"
        "- **Patterns** — interesting insights\n"
        "- **What to predict** — target variable suggestions"
    )

# ============================================================================
# SETTINGS PAGE
# ============================================================================
def render_settings():
    st.markdown("# ⚙️ Settings")
    
    st.markdown("### 🤖 LLM Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("Enable Local LLM", value=os.getenv("ENABLE_LOCAL_LLM", "").lower() == "true")
        st.checkbox("Auto-download LLM Model", value=os.getenv("AUTO_DOWNLOAD_LLM", "").lower() == "true")
    
    with col2:
        st.checkbox("Enable SHAP Explanations", value=True)
        st.checkbox("Enable Prometheus Metrics", value=os.getenv("ENABLE_PROMETHEUS", "").lower() == "true")
    
    st.markdown("---")
    
    st.markdown("### 📊 Analysis Settings")
    
    st.slider("Max rows for first pass", 1000, 100000, 25000, step=1000)
    st.slider("Model time budget (seconds)", 10, 300, 60, step=10)
    
    st.markdown("---")
    
    st.markdown("### 📁 Data Directories")
    st.text_input("Local Data Directory", value=os.getenv("LOCAL_DATA_DIR", "local_data"))
    st.text_input("Log Directory", value=os.getenv("LOG_DIR", "logs"))
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.info("""
    **Assay Analytics** v1.0
    
    An AI-powered data analysis platform featuring:
    - Automatic model selection
    - Semantic data enrichment
    - Local LLM integration
    - SHAP explanations
    
    Built with Streamlit, scikit-learn, and ❤️
    """)

# ============================================================================
# MAIN ROUTING
# ============================================================================
def main():
    page = st.session_state.page
    
    if page == "home":
        render_home()
    elif page == "analyze":
        render_analyze()
    elif page == "results":
        render_results()
    elif page == "chat":
        render_chat()
    elif page == "settings":
        render_settings()
    else:
        render_home()

if __name__ == "__main__":
    main()
