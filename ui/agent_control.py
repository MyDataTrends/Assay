"""
Agent Control Panel - UI for managing and monitoring Assay agents.
"""
import streamlit as st
import pandas as pd
import time
import os
import sys
from datetime import datetime
from pathlib import Path

# --- Imports for Functionality ---
from agents.config import load_agent_configs, get_agent_config
from agents.cli import _create_agent, _register_agents, _AGENT_CLASSES
from agents.memory.operational import OperationalMemory
from modeling.llm_dynamic_analyzer import dynamic_analyze as run_workflow

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"

def render_agent_control():
    """Render the main agent control interface."""
    
    # Custom CSS for the "Command Center" feel
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #0e1117;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #262730;
            border-bottom: 2px solid #ff4b4b;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Assay Ops Center")
    
    tab_fleet, tab_cmd, tab_academy = st.tabs([
        "🤖 Agent Fleet", 
        "🕹️ Command Center", 
        "🎓 Academy"
    ])

    # ==============================================================================
    # TAB 1: AGENT FLEET (Existing + Polish)
    # ==============================================================================
    with tab_fleet:
        st.caption("Monitor status, trigger runs, and view agent history.")

        # Ensure agents are registered
        _register_agents()
        configs = load_agent_configs()
        
        # helper to get agent status emoji
        def get_status_emoji(enabled):
            return "✅" if enabled else "⏸️"

        col1, col2 = st.columns([3, 1])
        with col1:
            agent_names = sorted(list(_AGENT_CLASSES.keys()))
            selected_agent = st.selectbox("Select Agent", agent_names, format_func=lambda x: f"{x.title()}")
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        if selected_agent:
            config = configs.get(selected_agent, get_agent_config(selected_agent))
            memory = OperationalMemory(selected_agent)
            
            # 0. Operations (Moved up for state access)
            col_run, col_opts = st.columns([1, 3])
            with col_opts:
                dry_run_override = st.checkbox("Force Dry Run", value=True, help="Run without making persistent changes")

            # 1. Status Card
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Status", "Enabled" if config.enabled else "Disabled", delta=get_status_emoji(config.enabled), delta_color="off")
                
                # Calculate effective mode
                is_dry = config.dry_run or dry_run_override
                mode_label = "Dry Run" if is_dry else "Live Execution"
                mode_delta = "SAFE" if is_dry else "ACTIVE"
                c2.metric("Mode", mode_label, delta=mode_delta, delta_color="normal" if is_dry else "inverse")
                
                c3.metric("Schedule", config.schedule.title())
                
                # Get last run stats
                history = memory.get_run_history(limit=1)
                last_run = history[0] if history else None
                last_run_time = "Never"
                if last_run:
                    try:
                        dt = datetime.fromisoformat(last_run["timestamp"])
                        last_run_time = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        last_run_time = last_run["timestamp"]
                c4.metric("Last Run", last_run_time)

            # 2. Input for Agent Instructions
                
            # Input for Agent Instructions
            instructions = st.text_area("Instructions / Context (Optional)", 
                                      help="Specific topic (e.g. 'Crypto') for Productizer, or general context for other agents.")

            with col_run:
                if st.button(f"▶️ Run {selected_agent.title()}", type="primary", use_container_width=True):
                    with st.status(f"Running {selected_agent}...", expanded=True) as status:
                        try:
                            status.write("🚀 Initializing agent...")
                            agent = _create_agent(selected_agent, dry_run_override=dry_run_override)
                            
                            status.write("🏃 Executing workflow...")
                            start_time = time.time()
                            
                            # Pass instructions to agent
                            run_kwargs = {}
                            if instructions:
                                if selected_agent == "productizer":
                                    run_kwargs["topic"] = instructions
                                else:
                                    run_kwargs["instructions"] = instructions
                            
                            result = agent.run(**run_kwargs)
                            duration = time.time() - start_time
                            
                            if result.success:
                                status.update(label=f"✅ Success ({duration:.1f}s)", state="complete", expanded=False)
                                st.success(result.summary)
                                if result.actions_taken:
                                    st.write("**Actions:**", result.actions_taken)
                            else:
                                status.update(label="❌ Failed", state="error", expanded=True)
                                st.error(f"Error: {result.error}")
                                
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            status.update(label="❌ Error", state="error")
                            st.error(f"System Error: {e}")

            # 3. History
            st.divider()
            st.subheader("📜 History & Activity")
            tab_h, tab_e = st.tabs(["Run History", "Escalations"])
            
            with tab_h:
                runs = memory.get_run_history(limit=10)
                if runs:
                    data = []
                    for r in runs:
                        data.append({
                            "Date": r["timestamp"],
                            "Status": "✅" if r["success"] else "❌",
                            "Duration": f"{r['duration']:.2f}s",
                            "Summary": r["summary"],
                            "Error": r["error"] or ""
                        })
                    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
                else:
                    st.info("No run history available.")

            with tab_e:
                escalations = memory.get_pending_escalations()
                if escalations:
                    for esc in escalations:
                        prio_map = {"urgent": "🔴", "review": "🟡", "fyi": "🟢", "metric": "📊"}
                        emoji = prio_map.get(esc["priority"], "❓")
                        with st.expander(f"{emoji} {esc['title']} ({esc['timestamp']})"):
                            st.write(esc["detail"])
                            if st.button("Resolve", key=f"res_{esc['id']}"):
                                 memory.resolve_escalation(esc['id'])
                                 st.success("Resolved!")
                                 time.sleep(0.5)
                                 st.rerun()
                else:
                    st.info("No pending escalations.")

    # ==============================================================================
    # TAB 2: COMMAND CENTER (New)
    # ==============================================================================
    with tab_cmd:
        st.header("🕹️ Command Center")
        st.caption("Direct control over data ingestion and analysis pipelines.")

        c1, c2 = st.columns(2)

        # --- Data Ingestion (Modernized) ---
        with c1:
            with st.container(border=True):
                st.subheader("📥 Data Ingestion")
                ingest_type = st.radio("Source", ["AI Auto-Discovery", "FRED (Economics)", "World Bank", "S3 Bucket"], horizontal=True)

                if ingest_type == "AI Auto-Discovery":
                    st.caption("Describe what you want, and the AI will find it.")
                    query = st.text_input("Data Description", placeholder="e.g. Steam top games, Spotify playlists, US Census population...", help="Natural language query")
                    
                    # Session state for auth retries
                    if "auth_retry_key" not in st.session_state:
                         st.session_state.auth_retry_key = None
                    
                    if st.button("🚀 Discover & Fetch", type="primary"):
                        try:
                            with st.spinner("🤖 AI is searching for APIs..."):
                                from mcp_server.discovery_agent import APIDiscoveryAgent
                                agent = APIDiscoveryAgent()
                                
                                # Use session key if available (retry flow)
                                api_key_to_use = st.session_state.get(f"api_key_input_{query}", None)
                                
                                result = agent.one_click_fetch(query, api_key=api_key_to_use)
                                
                                if result.success:
                                    if result.data:
                                        df = pd.DataFrame(result.data)
                                        # Normalize/Sanitize path
                                        safe_name = "".join(x for x in query if x.isalnum() or x in " _-").replace(" ", "_")
                                        save_path = DATASETS_DIR / f"AI_{safe_name}.csv"
                                        df.to_csv(save_path, index=False)
                                        st.success(f"{result.status}\nSaved to `datasets/AI_{safe_name}.csv` ({len(df)} rows)")
                                        st.dataframe(df.head(), hide_index=True)
                                    else:
                                        st.warning(result.status)
                                elif result.needs_auth:
                                    st.warning(f"🔒 {result.status}")
                                    st.info(f"The system found **{result.api_name}**, but it requires an API Key.")
                                    if result.api_id:
                                         st.caption(f"Registry ID: `{result.api_id}`")
                                    
                                    # We can't easily block execution here in waiting for input, 
                                    # so we show the input and ask user to click a "Retry with Key" button 
                                    # which effectively re-runs this block but we need to persist the state.
                                    # Actually, Streamlit reruns the whole script. 
                                    # So we should show the input below.
                                    st.session_state["last_auth_query"] = query
                                    st.session_state["last_auth_api"] = result.api_name
                                    
                                else:
                                    st.error(f"❌ {result.status}")
                                    
                        except Exception as e:
                            st.error(f"Discovery Failed: {e}")
                    
                    # Persistent Auth Input Block (Outside the button click to survive rerun? No, needs to be conditional)
                    # Actually, the best way in Streamlit is to check if we have a pending auth request for this query
                    if st.session_state.get("last_auth_query") == query:
                         st.divider()
                         st.markdown(f"#### 🔑 Enter Key for {st.session_state.get('last_auth_api')}")
                         key_input = st.text_input("API Key", type="password", key=f"key_input_{query}")
                         
                         if st.button("Authenticate & Retry"):
                             # This button trigger will re-run the script. 
                             # We need to capture the key from `key_input_{query}` and pass it to the agent.
                             # We can store it in a temp session var that the main block checks.
                             st.session_state[f"api_key_input_{query}"] = key_input
                             st.rerun()

                elif ingest_type == "FRED (Economics)":
                    st.caption("Sources: GDP, Unemployment, Inflation, etc.")
                    series_id = st.text_input("Series ID", "GDP", help="e.g. GDP, UNRATE, CPIAUCSL")
                    
                    if st.button("Fetch FRED Data", type="primary"):
                        try:
                            from public_data.connectors.fred import FREDConnector
                            with st.spinner("Connecting to Federal Reserve..."):
                                connector = FREDConnector()
                                df = connector.fetch_data(series_id)
                                if not df.empty:
                                    # Save to datasets
                                    save_path = DATASETS_DIR / f"FRED_{series_id}.csv"
                                    df.to_csv(save_path, index=False)
                                    st.success(f"Saved to `datasets/FRED_{series_id}.csv` ({len(df)} rows)")
                                else:
                                    st.warning("No data returned.")
                        except ImportError:
                            st.error("FRED Connector not available.")
                        except Exception as e:
                            st.error(f"Failed: {e}")

                elif ingest_type == "World Bank":
                    st.caption("Sources: Global development indicators")
                    indicator = st.text_input("Indicator", "NY.GDP.MKTP.KD.ZG", help="e.g. NY.GDP.MKTP.KD.ZG (GDP Growth)")
                    country = st.text_input("Country Code", "USA")
                    
                    if st.button("Fetch World Bank Data", type="primary"):
                        try:
                            from public_data.connectors.world_bank import WorldBankConnector
                            with st.spinner("Connecting to World Bank..."):
                                connector = WorldBankConnector()
                                df = connector.fetch_data(indicator, countries=country)
                                if not df.empty:
                                    save_path = DATASETS_DIR / f"WB_{indicator}_{country}.csv"
                                    df.to_csv(save_path, index=False)
                                    st.success(f"Saved to `datasets/WB_{indicator}_{country}.csv` ({len(df)} rows)")
                                else:
                                    st.warning("No data returned.")
                        except ImportError:
                            st.error("World Bank Connector not available.")
                        except Exception as e:
                            st.error(f"Failed: {e}")
                            
                elif ingest_type == "S3 Bucket":
                    bucket = st.text_input("Bucket Name")
                    prefix = st.text_input("Prefix", placeholder="raw/data/")
                    
                    if st.button("Sync S3", type="primary"):
                        if not bucket:
                            st.error("Bucket name is required.")
                        else:
                            with st.spinner("Syncing..."):
                                try:
                                    import boto3
                                    s3 = boto3.client("s3")
                                    # Basic sync logic since Data_Intake is deprecated
                                    objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
                                    count = 0
                                    for obj in objects:
                                        key = obj["Key"]
                                        if key.endswith("/"): continue
                                        filename = DATASETS_DIR / os.path.basename(key)
                                        s3.download_file(bucket, key, str(filename))
                                        count += 1
                                    st.success(f"Synced {count} files to `datasets/`")
                                except Exception as e:
                                    st.error(f"Failed: {e}")

        # --- Analysis Runner ---
        with c2:
            with st.container(border=True):
                st.subheader("🧠 Analysis Pipeline")
                
                # List files in datasets dir
                try:
                    files = [f.name for f in DATASETS_DIR.glob("*") if f.is_file() and f.suffix in [".csv", ".json", ".parquet", ".xlsx"]]
                except Exception:
                    files = []
                
                if not files:
                    st.warning(f"No data files found in {DATASETS_DIR}")
                    sel_file = None
                else:
                    sel_file = st.selectbox("Select Dataset", files)

                target_col = st.text_input("Target Column (Optional)", placeholder="e.g. Sales, Churn")
                
                if st.button("Run Analysis", type="primary", disabled=not sel_file):
                    with st.spinner("Running Analysis..."):
                        try:
                            # Load DF
                            file_path = DATASETS_DIR / sel_file
                            if sel_file.endswith(".csv"):
                                df = pd.read_csv(file_path)
                            elif sel_file.endswith(".json"):
                                df = pd.read_json(file_path)
                            elif sel_file.endswith(".parquet"):
                                df = pd.read_parquet(file_path)
                            elif sel_file.endswith(".xlsx"):
                                df = pd.read_excel(file_path)
                            else:
                                raise ValueError("Unsupported format")

                            result = run_workflow(df, target=target_col if target_col else None)
                            
                            st.success("Analysis Complete!")
                            with st.expander("Results", expanded=True):
                                st.json(result)
                                
                        except Exception as e:
                            st.error(f"Analysis Failed: {e}")

    # ==============================================================================
    # TAB 3: ACADEMY (New)
    # ==============================================================================
    with tab_academy:
        st.header("🎓 Assay Academy")
        
        st.markdown("""
        ### Welcome to the Agent Workforce!
        
        Assay uses a team of autonomous agents to keep your data platform healthy and insightful.
        Here is how they work.

        #### 🤖 Meet the Team

        | Agent | Role | Triggers |
        | :--- | :--- | :--- |
        | **Conductor** | The Boss. Orchestrates daily workflows and compiles the daily briefing. | Daily (Cron) |
        | **Engineer** | The Tech Lead. Scans code for debt and proposes refactors. | Manual / Weekly |
        | **Sentinel** | The Security Guard. Checks for vulnerabilities and test failures. | On-Event / Pre-Run |
        | **Advocate** | The Customer Success Manager. Handles user feedback. | On-Issue |
        
        #### 🚦 Escalation Levels
        
        Agents categorize their findings into 4 priority levels:
        
        *   🔴 **URGENT**: Needs your immediate attention (e.g., Security Vulnerability).
        *   🟡 **REVIEW**: Look at this when you have time (e.g., Code Smell, Optimizations).
        *   🟢 **FYI**: Informational only (e.g., "Daily backup successful").
        *   📊 **METRIC**: Data points for dashboards.

        #### ⌨️ CLI Reference
        
        Use these commands in your terminal for automation:

        ```bash
        # Start the platform
        assay serve           # Start API
        assay dashboard       # Start Main UI
        assay admin           # Start this Ops Center
        
        # Agents
        python -m agents run all
        python -m agents run conductor --dry-run
        ```
        """)
        
        with st.expander("Troubleshooting Guide"):
            st.markdown("""
            **Issue: Agents failing to start?**
            *   Check `logs/agents.log` for Python tracebacks.
            *   Ensure `requirements.txt` is installed.

            **Issue: Database locked?**
            *   Assay uses SQLite. Ensure no other heavy process is holding the `state/*.db` files.
            
            **Issue: Credentials missing?**
            *   Check your `.env` file for `OPENAI_API_KEY` or `AWS_ACCESS_KEY_ID`.
            """)
