import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd

from data_discovery.ckan_client import DataGovInClient
from data_discovery.dataset_catalog import DatasetCatalog
from data_processing.data_fetcher import DataFetcher
from data_processing.data_cleaner import DataCleaner
from data_processing.data_store import DataStore
from llm_integration.llm_client import LLMClient
from query_engine.query_parser import QueryParser
from query_engine.query_planner import QueryPlanner
from query_engine.query_executor import QueryExecutor
from config import get_config

load_dotenv()

st.set_page_config(page_title="Samarth Q&A System", layout="wide")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.data_store = None
    st.session_state.query_engine = None
    st.session_state.initializing = False

def initialize_system():
    """Initialize the entire system"""
    with st.spinner("Initializing system..."):
        # Step 1: Create components
        client = DataGovInClient()
        catalog = DatasetCatalog()
        fetcher = DataFetcher()
        cleaner = DataCleaner()
        data_store = DataStore()
        
        # Step 2: Discover datasets
        st.info("Discovering datasets from data.gov.in...")
        catalog.discover_datasets(client)
        
        # Step 3: Fetch and load key datasets
        st.info("Fetching and processing datasets...")
        for category, subcats in catalog.datasets.items():
            for subcat, info in subcats.items():
                for resource_info in info['resource_ids'][:2]:  # Load first 2 of each
                    try:
                        df = fetcher.fetch_dataset(resource_info['id'])
                        if not df.empty:
                            cleaned_df = cleaner.clean_dataset(df, category)
                            data_store.add_dataset(
                                category,
                                subcat,
                                cleaned_df,
                                resource_info
                            )
                    except Exception as e:
                        st.warning(f"Could not load {resource_info['name']}: {e}")

        # Step 3b: Explicitly ensure key agriculture resources are loaded by ID
        try:
            cfg = get_config()
            # a) All India Area Production Yield Major Crops
            if cfg.crop_production_resource_id:
                try:
                    df_major = fetcher.fetch_dataset(cfg.crop_production_resource_id)
                    if not df_major.empty:
                        cleaned_major = cleaner.clean_dataset(df_major, 'agriculture')
                        data_store.add_dataset(
                            'agriculture',
                            'crop_production_major_crops',
                            cleaned_major,
                            {'id': cfg.crop_production_resource_id, 'name': 'All India Area Production Yield Major Crops'}
                        )
                except Exception as e:
                    st.warning(f"Could not load major crops dataset: {e}")
            
            # b) District-wise Season-wise Crop Production (with auto-discovery fallback)
            district_resource_ids_to_try = []
            if cfg.district_crop_production_resource_id:
                district_resource_ids_to_try.append(cfg.district_crop_production_resource_id)
            
            # Try auto-discovery if configured ID fails or is missing
            try:
                # Use the client already created at the top of initialize_system
                discovered_district_id = client.discover_district_crop_production_resource_id()
                if discovered_district_id and discovered_district_id not in district_resource_ids_to_try:
                    district_resource_ids_to_try.append(discovered_district_id)
            except Exception:
                pass
            
            # Try each resource ID until we find a valid one
            district_loaded = False
            for resource_id in district_resource_ids_to_try:
                try:
                    df_district = fetcher.fetch_dataset(resource_id)
                    if not df_district.empty:
                        # Quick schema check - look for required columns
                        df_cols_lower = [c.lower() for c in df_district.columns]
                        has_state = any('state' in c for c in df_cols_lower)
                        has_district = any('district' in c for c in df_cols_lower)
                        has_crop = any('crop' in c for c in df_cols_lower)
                        has_production = any('production' in c or 'prodn' in c or 'prod' in c for c in df_cols_lower)
                        
                        if has_state and has_district and has_crop and has_production:
                            cleaned_district = cleaner.clean_dataset(df_district, 'agriculture')
                            data_store.add_dataset(
                                'agriculture',
                                'crop_production_district_season',
                                cleaned_district,
                                {'id': resource_id, 'name': 'District-wise Season-wise Crop Production'}
                            )
                            district_loaded = True
                            break
                except Exception as e:
                    # Try next resource ID
                    continue
            
            if not district_loaded and district_resource_ids_to_try:
                st.warning("Could not load district crop production dataset. The resource ID(s) may be incorrect or the dataset may be unavailable.")
        except Exception as e:
            st.warning(f"Could not ensure agriculture resources by ID: {e}")
        
        # Step 4: Initialize query engine
        llm_client = LLMClient()
        parser = QueryParser(llm_client)
        executor = QueryExecutor(data_store)
        planner = QueryPlanner(executor)
        
        st.session_state.data_store = data_store
        st.session_state.query_engine = {
            'parser': parser,
            'planner': planner,
            'llm': llm_client
        }
        st.session_state.initialized = True

def main():
    st.title("🌾Samarth Intelligent Q&A System")
    st.markdown("Ask questions about Indian agriculture and climate data")
    
    # Sidebar
    with st.sidebar:
        st.header("System Status")
        
        if st.session_state.initialized:
            st.success("✅ System Ready")
            
            st.header("Available Datasets")
            if st.session_state.data_store:
                datasets = st.session_state.data_store.list_datasets()
                for category, names in datasets.items():
                    st.subheader(category.title())
                    for name in names:
                        st.text(f"• {name}")
        else:
            # Auto-initialize on first load
            st.info("Preparing system...")
            if not st.session_state.initializing:
                st.session_state.initializing = True
                initialize_system()
                st.session_state.initializing = False
                st.success("System initialized!")
    
    # Main interface
    if st.session_state.initialized:
        # Query input
        question = st.text_area(
            "Ask your question:",
            placeholder="e.g., Compare average rainfall in Karnataka and Tamil Nadu for the last 5 years",
            height=100
        )
        
        if st.button("Submit Question", type="primary"):
            if question:
                with st.spinner("Processing query..."):
                    try:
                        # Parse query
                        parser = st.session_state.query_engine['parser']
                        available = st.session_state.data_store.list_datasets()
                        parsed = parser.parse_query(question, available)
                        
                        with st.expander("Show details: Query Analysis", expanded=False):
                            st.subheader("Query Analysis")
                            st.json(parsed)
                        
                        # Execute query
                        planner = st.session_state.query_engine['planner']
                        results = planner.plan_and_execute(parsed)
                        
                        with st.expander("Show details: Raw Query Results", expanded=False):
                            st.subheader("Query Results (raw)")
                            st.json(results)

                        # Check for unknown intent or errors
                        if parsed.get("intent") == "unknown":
                            st.warning("⚠️ **Query Not Recognized**")
                            st.info("Your query didn't match any of the supported patterns. Please try one of these formats:")
                            
                            suggestions = parsed.get("suggestions", [
                                "Compare rainfall in [State1] and [State2] for the last N years",
                                "List the top N crops produced in [State] during the last M years",
                                "Which district in [State] had the highest [Crop] production in [Year]",
                                "Compare [Crop] production across all districts in [State] for the last N years"
                            ])
                            
                            for i, suggestion in enumerate(suggestions, 1):
                                st.markdown(f"{i}. {suggestion}")
                            
                            st.markdown("---")
                            st.markdown(f"**Your question:** {parsed.get('question', question)}")
                            st.markdown("**Available datasets:**")
                            st.json(parsed.get('available_datasets', {}))
                        else:
                            # Structured visuals for supported result types
                            answer_data = results.get('answer_data', {}) or {}
                            cfg = get_config()

                            # Rainfall comparison chart
                            if 'rainfall_compare' in answer_data and isinstance(answer_data['rainfall_compare'], dict):
                                rc = answer_data['rainfall_compare']
                                if not rc.get('error') and isinstance(rc, dict):
                                    # Normalize payload (convert numpy types to Python types)
                                    def to_int_list(xs):
                                        try:
                                            return [int(x) for x in xs]
                                        except Exception:
                                            return xs

                                    # Build a dataframe and readable summary
                                    rows = []
                                    years_union = set()
                                    for state, payload in rc.items():
                                        if isinstance(payload, dict) and payload.get('avg_annual_mm') is not None:
                                            yrs = to_int_list(payload.get('years') or [])
                                            for y in yrs:
                                                years_union.add(y)
                                            rows.append({
                                                'State': state.title(),
                                                'Avg annual rainfall (mm)': float(payload['avg_annual_mm']),
                                                'Years covered': ', '.join(map(str, yrs)) if yrs else ''
                                            })
                                    if rows:
                                        # Show concise table
                                        st.markdown("**Rainfall comparison**")
                                        table_df = pd.DataFrame(rows)
                                        st.dataframe(table_df, width='stretch')

                                        # Bar chart for quick visual
                                        chart_df = table_df[['State', 'Avg annual rainfall (mm)']].set_index('State')
                                        st.bar_chart(chart_df)

                                        # Year range text
                                        if years_union:
                                            ymin, ymax = min(years_union), max(years_union)
                                            st.caption(f"Period: {ymin}–{ymax}")
                            # Source
                            if cfg.rainfall_resource_id:
                                st.caption(f"Source: data.gov.in Datastore • Resource: https://api.data.gov.in/resource/{cfg.rainfall_resource_id}")

                            # Top crops tables and charts
                            if 'top_crops' in answer_data and isinstance(answer_data['top_crops'], dict):
                                tc = answer_data['top_crops']
                                for state, records in tc.items():
                                    if isinstance(records, list) and records:
                                        st.markdown(f"**Top crops in {state.title()}**")
                                        df_tc = pd.DataFrame.from_records(records)
                                        st.dataframe(df_tc, width='stretch')
                                        # Optional simple chart if numeric column exists
                                        numeric_cols = [c for c in df_tc.columns if pd.api.types.is_numeric_dtype(df_tc[c])]
                                        label_col = [c for c in df_tc.columns if c.lower() in ('crop', 'crop_name', 'commodity')]
                                        if numeric_cols and label_col:
                                            plot_df = df_tc[[label_col[0], numeric_cols[0]]].set_index(label_col[0])
                                            st.bar_chart(plot_df)
                            # Source
                            if cfg.crop_production_resource_id:
                                st.caption(f"Source (Major crops): https://api.data.gov.in/resource/{cfg.crop_production_resource_id}")
                            if cfg.district_crop_production_resource_id:
                                st.caption(f"Source (District-season crops): https://api.data.gov.in/resource/{cfg.district_crop_production_resource_id}")

                            # District crop extrema comparison
                            if 'district_crop_extrema' in answer_data and isinstance(answer_data['district_crop_extrema'], dict):
                                de = answer_data['district_crop_extrema']
                                rows = []
                                for label in ('max', 'min'):
                                    item = de.get(label)
                                    if isinstance(item, dict) and item:
                                        rows.append({
                                            'Type': 'Highest' if label == 'max' else 'Lowest',
                                            'State': str(item.get('state', '')).title(),
                                            'District': item.get('district', ''),
                                            'Crop': item.get('crop', ''),
                                            'Year': item.get('year', ''),
                                            'Production': item.get('production', ''),
                                        })
                                if rows:
                                    st.markdown('**District production comparison (most recent year per subset)**')
                                    st.dataframe(pd.DataFrame(rows), width='stretch')
                                if cfg.district_crop_production_resource_id:
                                    st.caption(f"Source: https://api.data.gov.in/resource/{cfg.district_crop_production_resource_id}")

                            # Top N crops in a state
                            if 'top_crops_state' in answer_data and isinstance(answer_data['top_crops_state'], dict):
                                tcs = answer_data['top_crops_state']
                                if tcs.get('error'):
                                    st.warning(f"⚠️ {tcs.get('error')}")
                                    if tcs.get('debug'):
                                        with st.expander("Debug Information"):
                                            st.json(tcs.get('debug'))
                                elif isinstance(tcs.get('crops'), list):
                                    state_name = tcs.get('state', '').title()
                                    top_n = tcs.get('top_n', 0)
                                    year_range = tcs.get('year_range', [])
                                    crops_data = tcs.get('crops', [])
                                    
                                    if crops_data:
                                        # Build dataframe with readable column names
                                        df_top = pd.DataFrame(crops_data)
                                        # Rename columns to be readable
                                        col_rename = {}
                                        for old_col in df_top.columns:
                                            if old_col.lower() in ['crop', 'crop_name', 'commodity']:
                                                col_rename[old_col] = 'Crop'
                                            elif old_col.lower() in ['production', 'prodn', 'prod']:
                                                col_rename[old_col] = 'Production (tonnes)'
                                        df_top = df_top.rename(columns=col_rename)
                                        
                                        st.markdown(f"**Top {top_n} crops in {state_name}**")
                                        if year_range:
                                            st.caption(f"Based on data from {min(year_range)}–{max(year_range)}")
                                        st.dataframe(df_top, width='stretch')
                                        
                                        # Show warning if fewer crops returned than requested
                                        if len(crops_data) < top_n:
                                            st.warning(f"⚠️ Only {len(crops_data)} crop(s) found in the dataset (requested {top_n}). This may indicate the dataset doesn't have state-level breakdown or has limited crop data.")
                                            if tcs.get('debug_info'):
                                                with st.expander("Debug Information"):
                                                    st.json(tcs.get('debug_info'))
                                        
                                        # Chart if we have production column
                                        prod_col = [c for c in df_top.columns if 'production' in c.lower()]
                                        crop_col = [c for c in df_top.columns if 'crop' in c.lower()]
                                        if prod_col and crop_col:
                                            chart_df = df_top[[crop_col[0], prod_col[0]]].set_index(crop_col[0])
                                            st.bar_chart(chart_df)
                                        
                                        # Source attribution
                                        if cfg.crop_production_resource_id:
                                            st.caption(f"Source: https://api.data.gov.in/resource/{cfg.crop_production_resource_id}")

                            # District highest crop in specific year
                            if 'district_highest_crop_year' in answer_data and isinstance(answer_data['district_highest_crop_year'], dict):
                                dhc = answer_data['district_highest_crop_year']
                                # Check for error first
                                if dhc.get('error'):
                                    st.warning(f"⚠️ {dhc.get('error', 'Unknown error')}")
                                    if dhc.get('warning'):
                                        st.info(f"ℹ️ {dhc.get('warning')}")
                                    if dhc.get('total_production'):
                                        # Show state-level fallback data
                                        st.markdown(f"**State-level {dhc.get('crop', '').title()} production in {dhc.get('state', '').title()} ({dhc.get('year', '')})**")
                                        result_df = pd.DataFrame([{
                                            'State': dhc.get('state', '').title(),
                                            'Crop': dhc.get('crop', '').title(),
                                            'Total Production (tonnes)': dhc.get('total_production', 0),
                                            'Year': dhc.get('year', ''),
                                        }])
                                        st.dataframe(result_df, width='stretch')
                                        st.caption("⚠️ Note: This is state-level data. District-level data is not available.")
                                    if dhc.get('suggestion'):
                                        st.info(dhc.get('suggestion'))
                                    if dhc.get('alternative_queries'):
                                        st.markdown("**Alternative queries you can try:**")
                                        for alt_q in dhc.get('alternative_queries', []):
                                            st.markdown(f"- {alt_q}")
                                elif not dhc.get('error') and dhc.get('district'):
                                    st.markdown(f"**Highest {dhc.get('crop', '').title()} production in {dhc.get('state', '').title()} ({dhc.get('year', '')})**")
                                    result_df = pd.DataFrame([{
                                        'District': dhc.get('district', ''),
                                        'Production (tonnes)': dhc.get('production', 0),
                                        'Year': dhc.get('year', ''),
                                    }])
                                    st.dataframe(result_df, width='stretch')
                                    if cfg.district_crop_production_resource_id:
                                        st.caption(f"Source: https://api.data.gov.in/resource/{cfg.district_crop_production_resource_id}")

                            # District crop comparison across all districts
                            if 'district_crop_comparison' in answer_data and isinstance(answer_data['district_crop_comparison'], dict):
                                dcc = answer_data['district_crop_comparison']
                                if not dcc.get('error') and isinstance(dcc.get('districts'), list):
                                    state_name = dcc.get('state', '').title()
                                    crop_name = dcc.get('crop', '').title()
                                    year_range = dcc.get('year_range', [])
                                    districts_data = dcc.get('districts', [])
                                    
                                    if districts_data:
                                        st.markdown(f"**{crop_name} production across districts in {state_name}**")
                                        if year_range:
                                            st.caption(f"Aggregated over {min(year_range)}–{max(year_range)} ({len(year_range)} years)")
                                        
                                        # Create dataframe with readable column names
                                        df_districts = pd.DataFrame(districts_data)
                                        col_rename = {}
                                        for old_col in df_districts.columns:
                                            if old_col.lower() in ['district', 'district_name']:
                                                col_rename[old_col] = 'District'
                                            elif old_col.lower() in ['production', 'prodn', 'prod']:
                                                col_rename[old_col] = 'Total Production (tonnes)'
                                        df_districts = df_districts.rename(columns=col_rename)
                                        df_districts = df_districts.sort_values(by=[c for c in df_districts.columns if 'production' in c.lower()][0], ascending=False)
                                        
                                        st.dataframe(df_districts, width='stretch')
                                        
                                        # Bar chart for top districts
                                        prod_col_display = [c for c in df_districts.columns if 'production' in c.lower()]
                                        district_col_display = [c for c in df_districts.columns if 'district' in c.lower()]
                                        if prod_col_display and district_col_display:
                                            # Show top 20 districts in chart
                                            chart_df = df_districts.head(20)[[district_col_display[0], prod_col_display[0]]].set_index(district_col_display[0])
                                            st.bar_chart(chart_df)
                                        
                                        if cfg.district_crop_production_resource_id:
                                            st.caption(f"Source: https://api.data.gov.in/resource/{cfg.district_crop_production_resource_id}")
                            
                            # Generate answer
                            llm = st.session_state.query_engine['llm']
                            from llm_integration.prompt_templates import PromptTemplates
                            templates = PromptTemplates()
                            answer_prompt = templates.answer_synthesis_prompt(question, results)
                            # Pass results directly for better error handling
                            answer = llm.generate_response(answer_prompt, results=results)
                            
                            st.subheader("Answer")
                            st.markdown(answer)
                        
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
            else:
                st.warning("Please enter a question")
    else:
        st.info("Initializing system... please wait a moment")
        
        # Sample questions
        st.subheader("Sample Questions")
        st.markdown("""
        - Compare average rainfall in Karnataka and Punjab for the last 5 years
        - Which district in Maharashtra has the highest rice production?
        - Analyze wheat production trend in Uttar Pradesh over the last decade
        - What are data-backed arguments for promoting millets in drought-prone regions?
        """)

if __name__ == "__main__":
    main()