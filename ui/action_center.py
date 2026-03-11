"""
Action Center UI - Frontend interface for Assay platform actions.

Provides visual controls for:
- Data enrichment with public datasets/APIs
- Analysis execution
- Data fetching from external sources
"""
import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path

# Lazy imports to avoid circular dependencies
def _get_column_meta(df: pd.DataFrame):
    from preprocessing.metadata_parser import infer_column_meta
    return infer_column_meta(df)


def _get_enrichment_suggestions(column_meta) -> List[Dict[str, Any]]:
    """
    Generate enrichment suggestions based on detected column roles.
    
    Returns list of suggested datasets/APIs that could enrich the data.
    """
    roles = {m.role for m in column_meta if m.role != "unknown"}
    role_to_column = {m.role: m.name for m in column_meta if m.role != "unknown"}
    
    suggestions = []
    
    # Economic data suggestions
    if "date" in roles or "datetime" in roles:
        date_col = role_to_column.get("date") or role_to_column.get("datetime")
        suggestions.append({
            "id": "fred_gdp",
            "name": "FRED: GDP & Economic Growth",
            "source": "Federal Reserve Economic Data",
            "match_column": date_col,
            "match_type": "date",
            "description": "Quarterly GDP, growth rates, economic indicators",
            "columns_added": ["gdp", "gdp_growth_rate", "recession_indicator"],
            "category": "economic",
        })
        suggestions.append({
            "id": "fred_unemployment",
            "name": "FRED: Employment Statistics",
            "source": "Bureau of Labor Statistics via FRED",
            "match_column": date_col,
            "match_type": "date",
            "description": "Unemployment rate, labor force participation",
            "columns_added": ["unemployment_rate", "labor_force_participation"],
            "category": "economic",
        })
        suggestions.append({
            "id": "fred_inflation",
            "name": "FRED: Inflation & CPI",
            "source": "Federal Reserve Economic Data",
            "match_column": date_col,
            "match_type": "date",
            "description": "Consumer Price Index, inflation rates",
            "columns_added": ["cpi", "inflation_rate", "core_inflation"],
            "category": "economic",
        })
    
    # Geographic data suggestions
    if "zip_code" in roles:
        zip_col = role_to_column["zip_code"]
        suggestions.append({
            "id": "census_demographics",
            "name": "Census: Demographics",
            "source": "US Census Bureau",
            "match_column": zip_col,
            "match_type": "zip_code",
            "description": "Population, age distribution, education levels",
            "columns_added": ["population", "median_age", "pct_college_educated"],
            "category": "demographics",
        })
        suggestions.append({
            "id": "census_income",
            "name": "Census: Income & Poverty",
            "source": "US Census Bureau ACS",
            "match_column": zip_col,
            "match_type": "zip_code",
            "description": "Median household income, poverty rates",
            "columns_added": ["median_income", "poverty_rate", "income_per_capita"],
            "category": "demographics",
        })
        suggestions.append({
            "id": "housing_prices",
            "name": "Housing Market Data",
            "source": "Zillow / FHFA",
            "match_column": zip_col,
            "match_type": "zip_code",
            "description": "Home values, rental prices, market trends",
            "columns_added": ["median_home_value", "median_rent", "home_price_yoy"],
            "category": "housing",
        })
    
    if "state" in roles:
        state_col = role_to_column["state"]
        suggestions.append({
            "id": "state_economics",
            "name": "State Economic Indicators",
            "source": "Bureau of Economic Analysis",
            "match_column": state_col,
            "match_type": "state",
            "description": "State GDP, employment, industry breakdown",
            "columns_added": ["state_gdp", "state_unemployment", "top_industry"],
            "category": "economic",
        })
    
    # Consumer data suggestions
    if "date" in roles:
        date_col = role_to_column.get("date") or role_to_column.get("datetime")
        suggestions.append({
            "id": "consumer_sentiment",
            "name": "Consumer Sentiment Index",
            "source": "University of Michigan",
            "match_column": date_col,
            "match_type": "date",
            "description": "Consumer confidence, spending expectations",
            "columns_added": ["consumer_sentiment", "buying_conditions"],
            "category": "consumer",
        })
        suggestions.append({
            "id": "retail_sales",
            "name": "Retail Sales Data",
            "source": "US Census Bureau",
            "match_column": date_col,
            "match_type": "date",
            "description": "Monthly retail sales by category",
            "columns_added": ["retail_sales_total", "retail_sales_yoy"],
            "category": "consumer",
        })
    
    # Stock market suggestions
    if "ticker" in roles or "symbol" in roles:
        ticker_col = role_to_column.get("ticker") or role_to_column.get("symbol")
        suggestions.append({
            "id": "stock_prices",
            "name": "Stock Price History",
            "source": "Alpha Vantage / Yahoo Finance",
            "match_column": ticker_col,
            "match_type": "ticker",
            "description": "Historical stock prices, volume, splits",
            "columns_added": ["close_price", "volume", "market_cap"],
            "category": "financial",
        })
        suggestions.append({
            "id": "company_financials",
            "name": "Company Financials",
            "source": "Financial Modeling Prep",
            "match_column": ticker_col,
            "match_type": "ticker",
            "description": "Revenue, earnings, P/E ratios",
            "columns_added": ["revenue", "net_income", "pe_ratio"],
            "category": "financial",
        })
    
    # Global data
    if "country" in roles or "country_code" in roles:
        country_col = role_to_column.get("country") or role_to_column.get("country_code")
        suggestions.append({
            "id": "world_bank_development",
            "name": "World Bank: Development Indicators",
            "source": "World Bank Open Data",
            "match_column": country_col,
            "match_type": "country",
            "description": "GDP per capita, life expectancy, education",
            "columns_added": ["gdp_per_capita", "life_expectancy", "literacy_rate"],
            "category": "global",
        })
    
    return suggestions


def render_enrichment_panel(df: pd.DataFrame, meta=None):
    """Render the data enrichment section with suggestions."""
    
    # Load environment variables for API keys
    try:
        from utils.env_loader import load_env
        load_env()
    except ImportError:
        pass
    
    if meta is None:
        meta = _get_column_meta(df)
    
    suggestions = _get_enrichment_suggestions(meta)
    
    if not suggestions:
        st.info("📊 No enrichment suggestions available. Upload data with recognizable columns like dates, zip codes, or state codes to see suggestions.")
        return
    
    # Auto-enrichment toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"Found **{len(suggestions)}** potential enrichments for your data:")
    with col2:
        auto_enrich = st.checkbox("Auto-enrich", value=False, 
                                   help="Automatically apply selected enrichments without confirmation")
    
    # Store in session state
    if "enrichment_auto" not in st.session_state:
        st.session_state["enrichment_auto"] = False
    st.session_state["enrichment_auto"] = auto_enrich
    
    # Group by category
    categories = {}
    for s in suggestions:
        cat = s.get("category", "other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(s)
    
    category_icons = {
        "economic": "📈",
        "demographics": "👥",
        "housing": "🏠",
        "consumer": "🛒",
        "financial": "💹",
        "global": "🌍",
        "other": "📦",
    }
    
    selected_enrichments = []
    
    for category, items in categories.items():
        icon = category_icons.get(category, "📦")
        st.markdown(f"**{icon} {category.title()}**")
        
        for item in items:
            col1, col2, col3 = st.columns([0.5, 3, 1.5])
            
            with col1:
                selected = st.checkbox("Select", key=f"enrich_{item['id']}", value=False, label_visibility="collapsed")
                if selected:
                    selected_enrichments.append(item)
            
            with col2:
                st.markdown(f"**{item['name']}**")
                st.caption(f"Matches on `{item['match_column']}` • Adds: {', '.join(item['columns_added'][:3])}")
            
            with col3:
                with st.popover("ℹ️ Details"):
                    st.write(f"**Source:** {item['source']}")
                    st.write(f"**Description:** {item['description']}")
                    st.write(f"**Columns added:** {', '.join(item['columns_added'])}")
    
    st.divider()
    
    # Action button
    if selected_enrichments:
        if auto_enrich:
            st.success(f"✅ {len(selected_enrichments)} enrichment(s) will be applied automatically")
        else:
            if st.button(f"🔄 Enrich Data ({len(selected_enrichments)} selected)", type="primary"):
                enriched_df, report = perform_enrichment(df, selected_enrichments)
                
                if enriched_df is not None and len(enriched_df.columns) > len(df.columns):
                    st.session_state["enriched_data"] = enriched_df
                    st.success(f"✅ Enrichment complete! Added {len(enriched_df.columns) - len(df.columns)} columns.")
                    
                    # Show what was added
                    new_cols = set(enriched_df.columns) - set(df.columns)
                    st.write("**New columns:**", ", ".join(new_cols))
                    
                    # Preview
                    with st.expander("Preview enriched data"):
                        st.dataframe(enriched_df.head(10))
                    
                    # Show report
                    if report.get("details"):
                        with st.expander("Enrichment details"):
                            for detail in report["details"]:
                                status = "✅" if detail.get("success") else "❌"
                                st.write(f"{status} {detail.get('source', 'Unknown')}: {detail.get('message', '')}")
                else:
                    st.warning("No new data was added. Check your API keys or data compatibility.")
                    if report.get("details"):
                        for detail in report["details"]:
                            if not detail.get("success"):
                                st.error(f"❌ {detail.get('source')}: {detail.get('message')}")
    else:
        st.caption("Select one or more enrichments above, then click 'Enrich Data'")


def perform_enrichment(df: pd.DataFrame, enrichments: List[Dict]) -> tuple:
    """
    Actually perform data enrichment by calling APIs.
    
    Args:
        df: User's DataFrame
        enrichments: List of selected enrichment configs
        
    Returns:
        Tuple of (enriched_df, report)
    """
    import os
    
    result_df = df.copy()
    report = {"details": []}
    
    for enrich in enrichments:
        enrich_id = enrich["id"]
        match_col = enrich["match_column"]
        
        try:
            # FRED enrichments
            if enrich_id.startswith("fred_"):
                from public_data.connectors.fred import FREDConnector
                fred = FREDConnector()
                
                if not fred.api_key:
                    report["details"].append({
                        "source": enrich["name"],
                        "success": False,
                        "message": "FRED API key not set. Add FRED_API_KEY to .env file."
                    })
                    continue
                
                # Determine which series to fetch
                series_map = {
                    "fred_gdp": ["GDP"],
                    "fred_unemployment": ["UNRATE"],
                    "fred_inflation": ["CPIAUCSL"],
                    "consumer_sentiment": ["UMCSENT"],
                    "retail_sales": ["RSXFS"],
                }
                
                series_ids = series_map.get(enrich_id, ["GDP"])
                
                # Get date range from user data
                if match_col in result_df.columns:
                    dates = pd.to_datetime(result_df[match_col])
                    start = dates.min().strftime("%Y-%m-%d")
                    end = dates.max().strftime("%Y-%m-%d")
                    
                    # Fetch data
                    fred_data = fred.fetch_multiple(series_ids, start, end)
                    
                    if not fred_data.empty:
                        # Merge on date
                        result_df[match_col] = pd.to_datetime(result_df[match_col])
                        fred_data["date"] = pd.to_datetime(fred_data["date"])
                        
                        # Use merge_asof for approximate date matching
                        result_df = result_df.sort_values(match_col)
                        fred_data = fred_data.sort_values("date")
                        
                        result_df = pd.merge_asof(
                            result_df, 
                            fred_data, 
                            left_on=match_col, 
                            right_on="date",
                            direction="nearest"
                        )
                        
                        report["details"].append({
                            "source": enrich["name"],
                            "success": True,
                            "message": f"Added {len(series_ids)} column(s) from FRED"
                        })
                    else:
                        report["details"].append({
                            "source": enrich["name"],
                            "success": False,
                            "message": "No data returned from FRED API"
                        })
            
            # Census enrichments
            elif enrich_id.startswith("census_"):
                from public_data.connectors.census import CensusConnector
                census = CensusConnector()
                
                # Fetch ZIP-level data
                if enrich["match_type"] == "zip_code":
                    census_data = census.fetch_by_zip()
                    
                    if not census_data.empty and "zip_code" in census_data.columns:
                        # Standardize ZIP codes
                        result_df["_zip_str"] = result_df[match_col].astype(str).str.zfill(5)
                        census_data["zip_code"] = census_data["zip_code"].astype(str).str.zfill(5)
                        
                        result_df = result_df.merge(
                            census_data,
                            left_on="_zip_str",
                            right_on="zip_code",
                            how="left"
                        )
                        result_df = result_df.drop(columns=["_zip_str"], errors="ignore")
                        
                        report["details"].append({
                            "source": enrich["name"],
                            "success": True,
                            "message": "Added Census demographics by ZIP"
                        })
                    else:
                        report["details"].append({
                            "source": enrich["name"],
                            "success": False,
                            "message": "Could not fetch Census ZIP data"
                        })
                else:
                    report["details"].append({
                        "source": enrich["name"],
                        "success": False,
                        "message": f"Match type '{enrich['match_type']}' not yet supported"
                    })
            
            # World Bank enrichments
            elif enrich_id == "world_bank_development":
                from public_data.connectors.world_bank import WorldBankConnector
                wb = WorldBankConnector()
                
                # Get unique countries
                if match_col in result_df.columns:
                    countries = result_df[match_col].unique()
                    country_str = ",".join(countries[:50])  # Limit to 50
                    
                    wb_data = wb.fetch_data("NY.GDP.PCAP.CD", countries=country_str)
                    
                    if not wb_data.empty:
                        # Get most recent year for each country
                        latest = wb_data.sort_values("year").groupby("country_code").last().reset_index()
                        
                        result_df = result_df.merge(
                            latest[["country_code", "ny_gdp_pcap_cd"]].rename(
                                columns={"ny_gdp_pcap_cd": "gdp_per_capita"}
                            ),
                            left_on=match_col,
                            right_on="country_code",
                            how="left"
                        )
                        
                        report["details"].append({
                            "source": enrich["name"],
                            "success": True,
                            "message": "Added World Bank GDP per capita"
                        })
                    else:
                        report["details"].append({
                            "source": enrich["name"],
                            "success": False,
                            "message": "No World Bank data returned"
                        })
            
            # Placeholder for other enrichments
            else:
                report["details"].append({
                    "source": enrich["name"],
                    "success": False,
                    "message": f"Enrichment '{enrich_id}' not yet implemented"
                })
                
        except Exception as e:
            report["details"].append({
                "source": enrich["name"],
                "success": False,
                "message": f"Error: {str(e)}"
            })
    
    return result_df, report


def render_analysis_panel(df: pd.DataFrame):
    """Render the analysis controls section."""
    
    st.subheader("Analysis Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Target column selection
        columns = ["(Auto-detect)"] + list(df.columns)
        target = st.selectbox("Target Column", columns, 
                              help="Column to predict. Auto-detect looks for common target names.")
    
    with col2:
        # Analysis type
        analysis_types = [
            "Auto (Best Fit)",
            "Classification",
            "Regression", 
            "Clustering",
            "Time Series",
            "Descriptive Only"
        ]
        analysis_type = st.selectbox("Analysis Type", analysis_types)
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            use_llm = st.checkbox("Use LLM-powered analysis", value=True)
            quick_mode = st.checkbox("Quick mode (faster, less thorough)", value=False)
        with col2:
            max_features = st.slider("Max features to use", 5, 50, 20)
            test_split = st.slider("Test split %", 10, 40, 20)
    
    # Run button
    if st.button("🧠 Run Analysis", type="primary"):
        with st.spinner("Running analysis..."):
            try:
                from modeling.llm_dynamic_analyzer import dynamic_analyze as run_workflow
                
                target_col = None if target == "(Auto-detect)" else target
                result = run_workflow(df, target=target_col)
                
                st.session_state["result"] = result
                st.success("Analysis complete! See results in the main view.")
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {e}")


def render_fetch_panel():
    """Render the external data fetching section."""
    
    st.subheader("Fetch External Data")
    
    source_type = st.radio("Source Type", ["API Endpoint", "S3 Bucket", "URL/File"], horizontal=True)
    
    if source_type == "API Endpoint":
        st.text_input("API Base URL", placeholder="https://api.example.com/data")
        
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("API Key (optional)", type="password")
        with col2:
            st.selectbox("Format", ["JSON", "CSV", "XML"])
        
        st.text_area("Query Parameters (JSON)", placeholder='{"series": "GDP", "start": "2020-01-01"}')
        
    elif source_type == "S3 Bucket":
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Bucket Name", placeholder="my-data-bucket")
        with col2:
            st.text_input("Key Prefix", placeholder="data/")
        
    else:  # URL/File
        st.text_input("URL", placeholder="https://example.com/data.csv")
    
    if st.button("📥 Fetch Data"):
        st.info("Data fetching will be implemented with API connectors.")


def render_action_center(df: pd.DataFrame, meta=None):
    """
    Main entry point for the Action Center tab.
    
    Provides visual controls for all major Assay operations:
    - Data enrichment with suggestions
    - Analysis configuration and execution
    - External data fetching
    - Natural language queries
    """
    
    st.header("🎯 Action Center")
    st.caption("Configure and execute data operations from one place")
    
    # Import NL query panels
    try:
        from ui.nl_query import render_nl_query_panel, render_chart_from_description
        nl_available = True
    except ImportError:
        nl_available = False
    
    # Main sections as expanders
    if nl_available:
        with st.expander("⚡ Quick Query", expanded=True):
            render_nl_query_panel(df)
        
        with st.expander("📊 Describe a Chart", expanded=False):
            render_chart_from_description(df)
    
    with st.expander("🔄 Enrich Your Data", expanded=False):
        render_enrichment_panel(df, meta)
    
    with st.expander("🧠 Run Analysis", expanded=False):
        render_analysis_panel(df)
    
    with st.expander("📥 Fetch External Data", expanded=False):
        render_fetch_panel()


# For testing standalone
if __name__ == "__main__":
    import numpy as np
    
    # Create test data
    test_df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=100, freq="D"),
        "zip_code": np.random.choice(["10001", "90210", "60601", "30301"], 100),
        "sales": np.random.uniform(100, 1000, 100),
        "state": np.random.choice(["NY", "CA", "IL", "GA"], 100),
    })
    
    st.set_page_config(page_title="Action Center", layout="wide")
    render_action_center(test_df)
