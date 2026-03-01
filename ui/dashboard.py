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

# Open with buffering=1 (line buffered) or flush immediately
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

# === Debug & crash logging (optional) ===
import faulthandler
faulthandler.enable()
# Note: logging.basicConfig already called at top of file

# Tell Streamlit to emit DEBUG logs
import os
os.environ.setdefault("STREAMLIT_SERVER_ENABLECORS", "true")
log_profile("Importing streamlit...")
import streamlit as st

# Import report generation
from reports.report_generator import generate_report_bytes

# Temporary cache clear to resolve unhashable dict error from previous sessions
if "cache_cleared" not in st.session_state:
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state["cache_cleared"] = True
    except Exception:
        pass

log_profile("Importing orchestration...")
from orchestration.orchestrate_workflow import run_workflow, orchestrate_workflow
log_profile("Importing chatbot...")
from chatbot.chatbot import chatbot_interface
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
log_profile("Importing orchestrator core...")
from orchestration.orchestrator import orchestrate_dashboard
from orchestration.data_quality_scorer import summarize_for_display
from ui import redaction_banner
from ui.exploratory_tab import render_exploratory_tab
from ui.action_center import render_action_center
from ui.llm_settings import render_llm_settings, render_llm_settings_compact
from reports.report_generator import generate_report_bytes

# Unified session context - must be imported after streamlit
from ui.session_context import get_context, migrate_legacy_state

log_profile("Imports complete!")

# Migrate any legacy session state keys to unified keys
migrate_legacy_state()


def _hash_df(df: pd.DataFrame) -> str:
    """Return a SHA1 hash for the contents of ``df``."""
    data = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def _available_identifiers(df: pd.DataFrame, dest_dir: str = "metadata") -> list[str]:
    """Return identifiers for saved metadata versions of ``df``."""
    h = _hash_df(df)
    path = Path(dest_dir)
    ids = []
    pattern = f"*_{h}_roles.json"
    for p in path.glob(pattern):
        stem = p.stem
        if stem.count("_") >= 2:
            ids.append(stem.split("_")[0])
    return sorted(set(ids))


def _analysis_suggestions(df: pd.DataFrame, meta) -> list[str]:
    """Return analysis suggestions based on LLM or column metadata."""
    try:
        # SKIP LLM on startup to prevent 30s hang
        # rec = recommend_models_with_llm(df)
        rec = "" 
    except Exception:
        rec = ""
    suggestions: list[str] = []
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

# Mock dataset for demonstration
mock_data = pd.DataFrame(
    {
        "Date": pd.date_range(start="2022-01-01", periods=100, freq="ME"),
        "Sales": [i * 1.05 for i in range(100)],
    }
)


# Mock visualization functions
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
    """Run dashboard orchestration based on computation results."""
    model_output = {
        "predictions": result.get("model_info", {}).get("predictions", []),
        "confidence_score": result.get("model_info", {}).get("confidence_score", 80),
    }
    model_type = (
        "time-series" if data is not None and any("date" in col.lower() for col in data.columns) else "generic"
    )
    try:
        return orchestrate_dashboard(
            data,
            model_output,
            model_type,
            target_column,
            {"general": {}},
            "",
        )
    except RuntimeError:
        return {}


# Mock model functions for scenarios
def generate_scenario(params):
    """
    Simulates a scenario based on the provided parameters.

    Args:
        params (dict): Parameters for the scenario.

    Returns:
        str: A summary of the simulated scenario.
    """
    return f"Simulated scenario with adjustments: {params}"
# ────────────────────────────────────────────────────────────────────────

# Main dashboard logic

st.title("Business Insights Dashboard")
st.write("Welcome to the interactive dashboard with chatbot support!")
redaction_banner()

# Display past run ids if available
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

if run_history:
    st.sidebar.subheader("Run History")
    for idx, entry in enumerate(run_history):
        rid = entry.get("run_id")
        if not rid:
            continue
        st.sidebar.write(rid)
        if st.sidebar.button("Rerun", key=f"rerun_{rid}_{idx}"):
            res = orchestrate_workflow(rid, rid, load_datalake_dfs())
            st.session_state["result"] = res
            st.rerun()

# Sidebar for data upload and selection

# LLM Model Status
try:
    from llm_manager.registry import get_registry
    _llm_registry = get_registry()
    _active_model = _llm_registry.get_active_model()
    if _active_model:
        st.sidebar.success(f"🤖 {_active_model.name}")
    else:
        st.sidebar.warning("🤖 No LLM selected")
except Exception:
    st.sidebar.info("🤖 LLM: Setup needed")

# Dev Mode Toggle
from ui.dev_settings import render_dev_toggle
render_dev_toggle()

st.sidebar.divider()

st.sidebar.divider()

# Secure Key Storage
try:
    from utils.key_storage import KeyStorage
    key_store = KeyStorage()
    
    # Auto-load keys into env
    for svc in key_store.list_services():
        key = key_store.get_key(svc)
        if key:
            # Map friendly service name to env var if possible
            # For now, we rely on the manual mapping or presets
            # Ideally, detailed mapping would be good. 
            # Simple hack: 
            if svc == "fred": os.environ["FRED_API_KEY"] = key
            if svc == "alphavantage": os.environ["ALPHAVANTAGE_API_KEY"] = key
            
            # OpenAI/Anthropic (if managed here later)
            if svc == "openai": os.environ["OPENAI_API_KEY"] = key
    
except Exception as e:
    st.sidebar.error(f"Key Store Error: {e}")
    key_store = None

st.sidebar.divider()

# === Data Manager ===
st.sidebar.divider()
st.sidebar.subheader("🗄️ Data Manager")

# Initialize datasets store
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}
if "primary_dataset_id" not in st.session_state:
    st.session_state["primary_dataset_id"] = None

# Show active datasets
if st.session_state["datasets"]:
    active_ds = st.sidebar.radio(
        "Primary Dataset", 
        options=list(st.session_state["datasets"].keys()),
        index=0 if not st.session_state["primary_dataset_id"] else list(st.session_state["datasets"].keys()).index(st.session_state["primary_dataset_id"])
    )
    st.session_state["primary_dataset_id"] = active_ds
    data = st.session_state["datasets"][active_ds]
    
    # Simple list of others (secondary)
    others = [k for k in st.session_state["datasets"].keys() if k != active_ds]
    if others:
        st.sidebar.caption(f"Secondary: {', '.join(others)}")
else:
    data = None
    st.sidebar.info("No datasets loaded")


with st.sidebar.expander("📥 Add Dataset", expanded=data is None):
    add_mode = st.radio("Source", ["Upload CSV", "Connect API", "Sample Data"], horizontal=True)

    if add_mode == "Upload CSV":
        uploaded_files = st.file_uploader("Upload CSV", type="csv", accept_multiple_files=True)
        if uploaded_files:
            for f in uploaded_files:
                name = f.name
                if name not in st.session_state["datasets"]:
                    with st.spinner(f"Loading {name}..."):
                        # Save to disk for Scheduler access
                        upload_dir = Path("User_Data/uploaded")
                        upload_dir.mkdir(parents=True, exist_ok=True)
                        file_path = upload_dir / name
                        with open(file_path, "wb") as buffer:
                            f.seek(0)
                            buffer.write(f.read())
                        
                        # Load into memory
                        from preprocessing.data_cleaning import standardize_dataframe
                        f.seek(0) # Reset pointer
                        df = pd.read_csv(f)
                        df = standardize_dataframe(df)
                        
                        st.session_state["datasets"][name] = df
                        
                        # Store path for Scheduler
                        if "dataset_paths" not in st.session_state:
                            st.session_state["dataset_paths"] = {}
                        st.session_state["dataset_paths"][name] = str(file_path.absolute())
                        
                        if not st.session_state["primary_dataset_id"]:
                            st.session_state["primary_dataset_id"] = name
            st.rerun()

    elif add_mode == "Connect API":
        st.caption("Load data directly from external sources")
        source = st.selectbox("Source", ["FRED (Econ)", "World Bank", "Custom / Generic API"], key="side_connect_source")
        
        if source == "FRED (Econ)":
            try:
                from public_data.connectors import FREDConnector
                fred = FREDConnector()
                series = [s.id for s in fred.get_available_series()]
                selected_series = st.multiselect("Select Series", series, default=["GDP"], key="side_fred_sel")
                if st.button("Load FRED Data", key="side_load_fred"):
                    with st.spinner("Fetching..."):
                        loaded_count = 0
                        for s_id in selected_series:
                            try:
                                df = fred.fetch_data(s_id)
                                if not df.empty:
                                    st.session_state["datasets"][f"FRED_{s_id}"] = df
                                    # Set last one as primary for immediate view
                                    st.session_state["primary_dataset_id"] = f"FRED_{s_id}"
                                    loaded_count += 1
                            except Exception as e:
                                st.error(f"⚠️ Could not load data for {s_id}. Please try again later. (Error: {e})")
                        
                        if loaded_count > 0:
                            st.success(f"Successfully loaded {loaded_count} datasets")
                            time.sleep(1)
                            st.rerun()
                            
            except ImportError:
                st.error("FRED connector missing")

        elif source == "World Bank":
             try:
                from public_data.connectors import WorldBankConnector
                wb = WorldBankConnector()
                indicators = [s.id for s in wb.get_available_series()]
                selected_inds = st.multiselect("Select Indicators", indicators, default=["NY.GDP.MKTP.KD.ZG"], key="side_wb_sel")
                country = st.text_input("Country Code", "USA", key="side_wb_country")
                
                if st.button("Load World Bank Data", key="side_load_wb"):
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
                                st.error(f"⚠️ Could not retrieve indicator {ind}. Check your connection. (Error: {e})")
                                
                        if loaded_count > 0:
                             st.success(f"Loaded {loaded_count} datasets")
                             time.sleep(1)
                             st.rerun()
             except ImportError:
                  st.error("World Bank connector missing")
                  
        elif source == "Custom / Generic API":
            api_url = st.text_input("API URL", "https://api.coindesk.com/v1/bpi/currentprice.json")
            method = st.selectbox("Method", ["GET", "POST"])
            headers_str = st.text_area("Headers (JSON)", "{}")
            data_key = st.text_input("Data Key (JSON path to entries)", value=None, help="If response is {data: [...]}, enter 'data'")
            
            if st.button("Fetch Custom API"):
                try:
                    import requests
                    import json
                    headers = json.loads(headers_str)
                    
                    with st.spinner(f"Fetching {api_url}..."):
                        if method == "GET":
                            resp = requests.get(api_url, headers=headers)
                        else:
                            resp = requests.post(api_url, headers=headers)
                            
                        resp.raise_for_status()
                        data_json = resp.json()
                        
                        # Extract list if key provided
                        if data_key and data_key in data_json:
                            data_json = data_json[data_key]
                            
                        # Convert to DataFrame
                        if isinstance(data_json, dict):
                            # Try to normalize
                            df = pd.json_normalize(data_json)
                        elif isinstance(data_json, list):
                            df = pd.DataFrame(data_json)
                        else:
                            st.error("Could not parse response into DataFrame")
                            df = pd.DataFrame()
                            
                        if not df.empty:
                            from preprocessing.data_cleaning import standardize_dataframe
                            df = standardize_dataframe(df)
                            name = "API_Custom_Data"
                            st.session_state["datasets"][name] = df
                            st.session_state["primary_dataset_id"] = name
                            st.success("Loaded Custom Data")
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"⚠️ API request failed. Please check the URL and parameters. (Error: {e})")



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
             except:
                 st.error("Sample file missing")
                 # Fallback
                 st.session_state["datasets"]["Mock Sales"] = standardize_dataframe(mock_data)
                 st.session_state["primary_dataset_id"] = "Mock Sales"
                 st.rerun()

if data is None and not st.session_state["datasets"]:
    # === First Run / Welcome Wizard ===
    st.markdown("## 🚀 Welcome to Assay")
    st.markdown("""
    **Your AI Data Analyst is ready.** 
    
    Assay runs entirely on your machine. To get started, you need data.
    """)
    
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        if st.button("Load Sample Data", type="primary", width="stretch"):
            try:
                from preprocessing.data_cleaning import standardize_dataframe
                # Load the bundled sample data
                sample_path = Path("datasets/sample_sales_data.csv")
                if sample_path.exists():
                    df = pd.read_csv(sample_path)
                    df = standardize_dataframe(df)
                    st.session_state["datasets"]["Sample Sales"] = df
                    st.session_state["primary_dataset_id"] = "Sample Sales"
                    st.success("Loaded Sample Sales Data! analysing...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Sample data not found at {sample_path}")
            except Exception as e:
                st.error(f"Failed to load sample: {e}")

    with c2:
        st.info("👈 Or use the sidebar to upload your own CSV!")

    # Stop execution here so we don't render empty tabs
    st.stop()
    
    # Fallback only if we didn't stop (shouldn't happen)
    data = mock_data

# Retrieve any stored metadata
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

# Main content tabs
main_tab, explore_tab, fabric_tab, action_tab, chat_tab, scheduler_tab, llm_tab = st.tabs([
    "📊 Data Preview", "🔍 Explore", "🕸️ Data Fabric", "⚡ Actions", "💬 Chat", "📅 Scheduler", "⚙️ LLM Settings"
])

with main_tab:
    st.dataframe(preview)

with explore_tab:
    try:
        # Get metadata for the current dataset
        dataset_id = st.session_state.get("primary_dataset_id")
        current_meta = st.session_state.get("dataset_metadata", {}).get(dataset_id, {})
        
        # Include dataset name in context - helps LLM infer data domain
        # e.g., "fred_data" implies Federal Reserve Economic Data
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
        st.error(f"⚠️ We ran into a hiccup analyzing this dataset. Try reloading the page. (Technical details: {e})")
        st.info("Try refreshing the page or uploading a different dataset.")

with fabric_tab:
    try:
        from ui.data_fabric import render_data_fabric_tab
        # Convert meta list to dicts for easier processing
        meta_dicts = [{"original_name": m.name, "semantic_type": getattr(m, 'semantic_type', None)} for m in meta] if meta else []
        render_data_fabric_tab(data, meta_dicts)
    except ImportError as e:
        st.warning(f"Data Fabric module not available: {e}")
    except Exception as e:
        st.error(f"⚠️ Data Fabric is temporarily unavailable. (Technical details: {e})")

with action_tab:
    try:
        render_action_center(data, meta)
    except Exception as e:
        st.error(f"⚠️ Action Center is temporarily unavailable. (Technical details: {e})")
        st.info("Try refreshing the page or check your data format.")


with scheduler_tab:
    try:
        from ui.scheduler_ui import render_scheduler_ui
        render_scheduler_ui()
    except ImportError:
        st.warning("Scheduler module not found.")
    except Exception as e:
        st.error(f"Scheduler error: {e}")

with llm_tab:
    render_llm_settings()
    
    # Learning Progress gamification widget
    st.divider()
    try:
        from ui.learning_progress import render_learning_progress
        render_learning_progress(compact=False)
    except ImportError:
        pass  # Module not yet available
        
    st.divider()
    
    # Teaching Mode Integration
    try:
        from ui.teach_logic import render_teaching_mode
        render_teaching_mode()
    except ImportError:
        st.info("Teaching module not loaded.")

# Suggest analyses based on data
if "analysis_suggestions" not in st.session_state:
    st.session_state["analysis_suggestions"] = _analysis_suggestions(data, meta)

if st.session_state["analysis_suggestions"]:
    st.sidebar.subheader("Suggested Analyses")
    for idx, sugg in enumerate(st.session_state["analysis_suggestions"]):
        if st.sidebar.button(sugg, key=f"sugg_{idx}"):
            st.session_state["suggestion"] = sugg
            st.rerun()

models = {
    "scenario_generator": generate_scenario,
    "classification": lambda p: select_analyzer(data).run(data),
    "regression": lambda p: select_analyzer(data).run(data),
    "clustering": lambda p: select_analyzer(data).run(data),
    "anomaly_detection": lambda p: select_analyzer(data).run(data),
    "forecasting": lambda p: select_analyzer(data).run(data),
}

# Banner for low accuracy prompting column review
if st.session_state.get("needs_role_review") and not st.session_state.get("show_column_review"):
    st.warning("Accuracy is low. Clarify column meanings to improve?")
    if st.button("Review Columns"):
        st.session_state["show_column_review"] = True

if st.session_state.get("show_column_review"):
    new_roles = column_review(data, st.session_state.get("column_meta", []))
    if new_roles:
        st.session_state["user_roles"] = new_roles
        store_role_corrections(data, new_roles)

# Put integrated chat in the chat tab (using new chat_mode.py components)
with chat_tab:
    st.subheader("💬 Chat with Your Data")
    
    # Use unified session context for chat state
    ctx = get_context()
        

    
    # Import chat mode functions and logger
    try:
        from ui.chat_logic import (
            detect_intent, generate_visualization_code, generate_analysis_code, generate_informational_response,
            safe_execute, safe_execute_viz, generate_natural_answer, 
            fallback_visualization, is_llm_ready, execute_analysis_with_retry
        )
        from llm_learning.interaction_logger import get_interaction_logger, InteractionType
        interaction_logger = get_interaction_logger()
        chat_available = True
    except ImportError as e:
        chat_available = False
        interaction_logger = None
        st.warning(f"Chat functionality not available: {e}")
    
    # First-run tutorial for LLM Learning
    if "seen_learning_tutorial" not in st.session_state:
        st.session_state.seen_learning_tutorial = False
    
    if not st.session_state.seen_learning_tutorial:
        with st.expander("🎓 **New! Your LLM Gets Smarter Over Time**", expanded=True):
            st.markdown("""
            **Assay now learns from your interactions!**
            
            - 💬 **Chat naturally** - Every question helps improve responses
            - ⭐ **Rate answers** - Your ratings train the model on what works
            - 📊 **Track progress** - See your LLM's learning journey in the **LLM Settings** tab
            
            The more you use Assay, the smarter it becomes for *your* data patterns!
            """)
            if st.button("Got it!", key="dismiss_tutorial"):
                st.session_state.seen_learning_tutorial = True
                st.rerun()
    
    if chat_available:
        llm_ready = is_llm_ready()
        
        if not llm_ready:
            st.warning("⚠️ LLM unavailable — configure one in the LLM Settings tab")
        
        # Build rich context from all available sources
        primary_id = st.session_state.get("primary_dataset_id", "")
        context_parts = []
        
        # Add info for ALL loaded datasets
        datasets = st.session_state.get("datasets", {})
        if datasets:
            context_parts.append(f"Primary Dataset: {primary_id}")
            context_parts.append("Available Datasets and Schemas:")
            
            for ds_name, df in datasets.items():
                is_primary = "(Primary)" if ds_name == primary_id else ""
                schema_str = (
                    f"Dataset: {ds_name} {is_primary}\n"
                    f"Columns: {', '.join(df.columns)}\n"
                    f"Types: {df.dtypes.to_dict()}\n"
                    f"Sample:\n{df.head(3).to_string()}\n"
                    f"---"
                )
                context_parts.append(schema_str)
        
        # Include AI Summary if already generated
        ai_summary = st.session_state.get("ai_summary")
        if ai_summary:
            context_parts.append(f"AI Summary: {ai_summary}")
        # Include chart suggestions context if available
        chart_data = st.session_state.get("chart_suggestions", {})
        if "suggestions" in chart_data and chart_data["suggestions"]:
            sugs = [s.get("title", "") for s in chart_data["suggestions"][:3]]
            context_parts.append(f"Suggested visualizations: {', '.join(sugs)}")
        
        # Add recent chat history (last 5 messages) for context
        if ctx.chat_history:
            history_str = "\nRECENT CONVERSATION:\n"
            for msg in ctx.chat_history[-5:]:
                role = msg["role"].upper()
                content = msg["content"]
                history_str += f"{role}: {content}\n"
            context_parts.append(history_str)

        chat_context = ". ".join(context_parts)
        
        # Display chat history (shared across all tabs)
        for msg in ctx.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Check for metadata content (visualizations/dataframes)
                metadata = msg.get("metadata", {})
                if "data" in metadata and metadata["data"] is not None:
                    msg_data = metadata["data"]
                    if hasattr(msg_data, 'show'):  # Plotly figure
                        st.plotly_chart(msg_data, width="stretch")
                    elif isinstance(msg_data, pd.DataFrame):
                        st.dataframe(msg_data)
                    else:
                        st.write(msg_data)
        
        # Chat input
        if prompt := st.chat_input("Ask about your data..."):
            ctx.add_message("user", prompt)
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                if not llm_ready:
                    response = "⚠️ LLM not configured. Go to LLM Settings tab to set up a model."
                    st.markdown(response)
                    ctx.add_message("assistant", response)
                else:
                    # Glass Box UX: Show the user what's happening
                    with st.status("Processing Request...", expanded=True) as status:
                        status.write("🧠 Identifying intent...")
                        intent = detect_intent(prompt, context=chat_context)
                        
                        if intent == "visualization":
                            status.write("📊 Generating visualization code...")
                            code = generate_visualization_code(data, prompt, context=chat_context, datasets=st.session_state.get("datasets"))
                            fig = None
                            if code:
                                status.write("🎨 Rendering chart...")
                                success, fig, error = safe_execute_viz(code, data, datasets=st.session_state.get("datasets"))
                            if not fig:
                                status.write("⚠️ Standard rendering failed, trying fallback...")
                                fig = fallback_visualization(data, prompt)
                            
                            status.update(label="Visualization Ready!", state="complete", expanded=False)
                            
                            if fig:
                                st.markdown("Here's your visualization:")
                                st.plotly_chart(fig, width="stretch")
                                ctx.add_message("assistant", "Here's your visualization:", metadata={"data": fig})
                                # Log successful interaction
                                if interaction_logger:
                                    interaction_logger.log(
                                        prompt=prompt,
                                        response="Visualization created",
                                        interaction_type=InteractionType.VISUALIZATION,
                                        code_generated=code or "",
                                        execution_success=True,
                                        dataset_name=st.session_state.get("primary_dataset_id", "")
                                    )
                            else:
                                response = "I couldn't create that visualization. Try 'show a bar chart of X'."
                                st.markdown(response)
                                ctx.add_message("assistant", response)
                                # Log failed interaction
                                if interaction_logger:
                                    interaction_logger.log(
                                        prompt=prompt,
                                        response=response,
                                        interaction_type=InteractionType.VISUALIZATION,
                                        code_generated=code or "",
                                        execution_success=False,
                                        dataset_name=st.session_state.get("primary_dataset_id", "")
                                    )
                        
                        elif intent == "informational":
                            status.write("📚 Consulting knowledge base...")
                            response = generate_informational_response(prompt, context=chat_context)
                            status.update(label="Response Ready!", state="complete", expanded=False)
                            
                            if not response:
                                response = "⚠️ I'm unable to generate a response right now. Please check if the LLM is running correctly."
                            st.markdown(response)
                            ctx.add_message("assistant", response)
                            
                            # Log informational interaction
                            if interaction_logger:
                                interaction_logger.log(
                                    prompt=prompt,
                                    response=response,
                                    interaction_type=InteractionType.CHAT,
                                    dataset_name=st.session_state.get("primary_dataset_id", "")
                                )
                            
                        else:
                            status.write("🧮 Generating analysis code...")
                            success, result, code, error = execute_analysis_with_retry(data, prompt, context=chat_context, datasets=st.session_state.get("datasets"))
                            
                            if success:
                                status.update(label="Analysis Complete!", state="complete", expanded=False)
                                natural = generate_natural_answer(prompt, result)
                                response = natural or "Here's what I found:"
                                st.markdown(response)
                                if result is not None:
                                    if isinstance(result, pd.DataFrame):
                                        st.dataframe(result)
                                    else:
                                        st.write(result)
                                ctx.add_message("assistant", response, metadata={"data": result})
                                # Log successful analysis
                                if interaction_logger:
                                    interaction_logger.log(
                                        prompt=prompt,
                                        response=response,
                                        interaction_type=InteractionType.ANALYSIS,
                                        code_generated=code,
                                        execution_success=True,
                                        dataset_name=st.session_state.get("primary_dataset_id", "")
                                    )
                                
                                # Show the code that generated this result (Reproducibility)
                                with st.expander("🔍 View Analysis Logic"):
                                    st.code(code, language="python")
                                    st.download_button(
                                        label="Download Script",
                                        data=code,
                                        file_name="analysis_script.py",
                                        mime="text/x-python"
                                    )
                            else:
                                status.update(label="Analysis Failed", state="error", expanded=True)
                                response = f"Error: {error}"
                                st.markdown(response)
                                ctx.add_message("assistant", response)
                                # Log failed analysis
                                if interaction_logger:
                                    interaction_logger.log(
                                        prompt=prompt,
                                        response=response,
                                        interaction_type=InteractionType.ANALYSIS,
                                        code_generated=code,
                                        execution_success=False,
                                        dataset_name=st.session_state.get("primary_dataset_id", "")
                                    )
                    st.rerun()

    # Export Report Section
    st.divider()
    with st.expander("📄 Export Analysis Report", expanded=False):
        st.write("Download an interactive HTML report of this conversation, including all visualizations.")
        col1, col2 = st.columns([3, 1])
        with col1:
            from datetime import datetime
            default_title = f"Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            report_title = st.text_input("Report Title", value=default_title)
        
        with col2:
            if st.button("Prepare Download"):
                # Collect figures from chat history
                figs = []
                for msg in st.session_state.get("chat_history", []):
                    if "data" in msg and msg["data"] is not None:
                        # Check if it looks like a Plotly figure (has to_json or similar)
                        if hasattr(msg["data"], "to_json"):
                            figs.append(msg["data"])
                
                # Generate report
                try:
                    report_bytes = generate_report_bytes(
                        df=data,
                        result=None,
                        figures=figs,
                        chat_history=st.session_state.get("chat_history", []),
                        title=report_title
                    )
                    
                    st.session_state["ready_report"] = report_bytes
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")

        if "ready_report" in st.session_state:
            st.download_button(
                label="📥 Download Interactive HTML",
                data=st.session_state["ready_report"],
                file_name=f"assay_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html"
            )

# Rating widget
st.subheader("Rate the results")
rating = st.slider("Select a score", 1, 5, 3, key="rating_slider")
if st.button("Submit Rating", key="submit_rating"):
    store_rating(rating)
    st.success("Thank you for your feedback!")

# Display analysis results if available
result = st.session_state.get("result")
if result:
    target_col = result.get("model_info", {}).get("target") or (
        data.columns[-1] if data is not None and len(data.columns) else ""
    )
    
    # Display Data Quality Report (if diagnostics available)
    diagnostics = result.get("diagnostics", {})
    if diagnostics:
        quality_summary = summarize_for_display(diagnostics)
        with st.expander(f"{quality_summary['status_emoji']} Data Quality Report", expanded=quality_summary['status'] != 'good'):
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
                st.info("💡 Consider reviewing column roles or providing additional data to improve results.")
    
    dash = build_dashboard(data, result, target_col)
    if dash.get("chart_result") is not None:
        st.write(dash["chart_result"])
    st.header("Analysis Results")
    decision = result.get("modeling_decision")
    if decision and not decision.get("modeling_required", True):
        st.info(
            f"\u2139\ufe0f No predictive model was run. Reason: {decision.get('reasoning', '')}"
        )
    elif decision and decision.get("modeling_required"):
        st.info(
            f"Predictive modeling applied: {decision.get('modeling_type', '')}. Reason: {decision.get('reasoning', '')}"
        )
    if result.get("modeling_failed"):
        st.error("Modeling failed. Showing descriptive results only.")
        if result.get("failure_reason"):
            st.info(result["failure_reason"])
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

    # Anomalies are shown once
    st.sidebar.write("Anomalies", result.get("anomalies"))

    if result.get("summary"):
        st.subheader("Business Summary")
        st.write(result["summary"])


    rec_models = result.get("recommended_models")
    if rec_models:
        with st.sidebar.expander("Recommended Models"):
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


    # Premium-only feature
    dl_button = getattr(st.sidebar, "download_button", None)
    if callable(dl_button):
        dl_button(
            "Download One Pager (Premium)",
            result.get("summary", ""),
            file_name="one_pager.txt",
            disabled=not st.session_state.get("is_paid", False),
        )

    model_path = Path("best_model.pkl")
    if model_path.exists():
        with st.sidebar:
            st.download_button(
                "Download best_model.pkl",
                model_path.read_bytes(),
                file_name="best_model.pkl",
                mime="application/octet-stream",
            )
    
    # HTML Report Export
    try:
        report_bytes = generate_report_bytes(data, result, title="Assay Analysis Report")
        st.sidebar.download_button(
            "📄 Export Full Report (HTML)",
            report_bytes,
            file_name="analysis_report.html",
            mime="text/html",
        )
    except Exception as e:
        st.sidebar.caption(f"Report generation unavailable")
            
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


