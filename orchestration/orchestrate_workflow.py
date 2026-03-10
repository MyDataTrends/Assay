from __future__ import annotations
from __future__ import annotations
from pathlib import Path

from storage.get_backend import backend
from modeling.model_selector import evaluate_model
from modeling.model_training import train_model, save_model, load_model
from preprocessing.llm_summarizer import generate_summary
from storage.local_backend import load_run_metadata
from orchestration.data_preprocessor import DataPreprocessor
from preprocessing.metadata_parser import parse_metadata, pre_scan_metadata
from config import (
    MAX_ROWS_FIRST_PASS,
    MAX_FEATURES_FIRST_PASS,
    PROFILE_SAMPLE_ROWS,
    MAX_REQUESTS_FREE,
    MAX_GB_FREE,
)
from utils.usage_tracker import check_quota

import logging
from utils.logging import configure_logging
import pandas as pd

from orchestration.workflow_manager import WorkflowManager
import json
from orchestration.orchestrator import orchestrate_dashboard

configure_logging()

import warnings

warnings.warn(
    "orchestrate_workflow is deprecated and will be removed in a future release. "
    "Please migrate to the Cascade Planner architecture.\n"
    "Migration Guide:\n"
    "- orchestrate_dashboard() -> orchestration.cascade_planner.cascade_execute()\n"
    "- metadata_parser heuristics -> orchestration.tool_registry._handle_data_profiler()\n"
    "- intent classification -> orchestration.classify_intent()",
    DeprecationWarning,
    stacklevel=2,
)

def orchestrate_workflow(
    user_id: str | None = None,
    file_name: str | None = None,
    datalake_dfs=None,
    target_column=None,
    category=None,
    user_labels: dict[str, str] | None = None,
    run_id: str | None = None,
    diagnostics_config: dict | None = None,
):
    """Entry point for running the workflow via ``WorkflowManager``."""
    df = None
    if datalake_dfs:
        df = next(iter(datalake_dfs.values()), None)
    elif user_id and file_name:
        try:
            dp = DataPreprocessor()
            df, _ = dp.load_dataset(user_id, file_name, run_id)
        except Exception as exc:  # pragma: no cover - safety
            logging.warning("Pre-scan failed: %s", exc)

    if df is not None:
        scan = pre_scan_metadata(df)
        if scan["rows"] > MAX_ROWS_FIRST_PASS or scan["columns"] > MAX_FEATURES_FIRST_PASS:
            logging.warning(
                "Dataset over caps - rows: %d cols: %d mem_mb: %.2f",
                scan["rows"],
                scan["columns"],
                scan["memory_bytes"] / (1024 ** 2),
            )
            sample = df.head(PROFILE_SAMPLE_ROWS)
            summary = parse_metadata(sample)
            return {"error": "dataset_over_cap", "summary": summary}

    manager = WorkflowManager(
        user_id or "",
        file_name or "",
        datalake_dfs,
        target_column=target_column,
        category=category,
        user_labels=user_labels,
        run_id=run_id,
    )
    result = manager.preprocess_and_cache(diagnostics_config)
    if result:
        return result
    manager.enrich_with_public_data()

    warning = None
    if user_id:
        allowed, msg, _ = check_quota(user_id)
        if not allowed:
            return msg or {"error": "quota_exceeded"}
        warning = msg.get("warning") if msg else None

    result = manager.select_and_run_analysis()
    if result:
        if warning:
            result["warning"] = warning
        return result

    if user_id:
        allowed, msg, _ = check_quota(user_id)
        if not allowed:
            return msg or {"error": "quota_exceeded"}
        if warning is None and msg:
            warning = msg.get("warning")

    manager.generate_outputs()
    if manager.result and warning:
        manager.result["warning"] = warning
    manager.trigger_agent_actions()
    return manager.result or {}

def run_workflow(data: pd.DataFrame, target: str = None):
    """
    UI-friendly shim around the core `orchestrate_workflow` engine.

    Uses LLM-powered dynamic analysis when available, with multiple fallback strategies.

    Args:
        data (pd.DataFrame): The input dataset for processing.
        target (str, optional): The target column name. Defaults to None.

    Returns:
        dict: The result of the workflow execution.
    """
    import uuid
    from datetime import datetime
    
    run_id = str(uuid.uuid4())[:8]
    errors_encountered = []
    
    # Strategy 1: Try LLM-powered dynamic analysis (most adaptive)
    try:
        from modeling.llm_dynamic_analyzer import dynamic_analyze
        logging.info("Attempting LLM-powered dynamic analysis")
        result = dynamic_analyze(data, target)
        
        if result and result.get('analysis_type') != 'error':
            # Enhance result with standard fields
            result['run_id'] = run_id
            result['timestamp'] = datetime.now().isoformat()
            result['rows'] = len(data)
            result['columns'] = len(data.columns)
            
            # Convert model info to serializable format
            if result.get('model'):
                result['model_info'] = {
                    'type': type(result['model']).__name__,
                    'explanations': result.get('feature_importance', {}),
                }
            
            logging.info(f"LLM dynamic analysis succeeded: {result.get('analysis_type')}")
            return result
        else:
            errors_encountered.append(f"LLM analysis returned error: {result.get('insights', [])}")
    except Exception as e:
        errors_encountered.append(f"LLM dynamic analysis failed: {e}")
        logging.warning(f"LLM dynamic analysis failed: {e}")
    
    # Strategy 2: Try standard orchestrate_workflow
    try:
        logging.info("Attempting standard orchestrate_workflow")
        user_id = "local_user"
        file_name = "local_upload.csv"
        user_dir = Path("local_data") / "User_Data" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        csv_path = user_dir / file_name
        data.to_csv(csv_path, index=False)

        result = orchestrate_workflow(
            user_id=user_id,
            file_name=file_name,
            datalake_dfs={file_name: data},
            target_column=target,
        )
        
        if result and not result.get('error'):
            logging.info("Standard workflow succeeded")
            return result
        else:
            errors_encountered.append(f"Standard workflow error: {result.get('error')}")
    except Exception as e:
        errors_encountered.append(f"Standard workflow failed: {e}")
        logging.warning(f"Standard workflow failed: {e}")
    
    # Strategy 3: Minimal fallback - just return descriptive stats
    try:
        logging.info("Using minimal fallback analysis")
        
        # Basic descriptive statistics
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        stats = data[numeric_cols].describe().to_dict() if numeric_cols else {}
        
        return {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'descriptive',
            'rows': len(data),
            'columns': len(data.columns),
            'column_names': list(data.columns),
            'dtypes': {col: str(data[col].dtype) for col in data.columns},
            'stats': stats,
            'metrics': {
                'completeness': (1 - data.isna().sum().sum() / data.size) * 100,
                'numeric_columns': len(numeric_cols),
            },
            'insights': [
                f"Dataset has {len(data):,} rows and {len(data.columns)} columns",
                f"Found {len(numeric_cols)} numeric columns",
                f"Data completeness: {(1 - data.isna().sum().sum() / data.size) * 100:.1f}%",
            ],
            'warnings': errors_encountered,
            'model_info': {},
        }
    except Exception as e:
        logging.error(f"Even minimal fallback failed: {e}")
        return {
            'run_id': run_id,
            'analysis_type': 'error',
            'error': str(e),
            'warnings': errors_encountered,
        }


def setup(*_args, **_kwargs):
    """Legacy no-op setup hook preserved for backward-compatible tests."""
    return None