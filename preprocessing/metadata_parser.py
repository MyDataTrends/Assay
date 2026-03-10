import pandas as pd
from config import PROFILE_MAX_COLS, PROFILE_SAMPLE_ROWS


def pre_scan_metadata(df: pd.DataFrame) -> dict:
    """Lightweight dataset scan to gather basic info.

    Computes row count, column types, null share per column, and estimated
    memory usage in bytes.
    """
    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
    return {
        "rows": len(df),
        "columns": len(dtypes),
        "dtypes": dtypes,
        "null_share": df.isna().mean().to_dict(),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }


def parse_metadata(df: pd.DataFrame) -> dict:
    """Extract basic metadata from a DataFrame."""
    limited = df.iloc[:PROFILE_SAMPLE_ROWS, :PROFILE_MAX_COLS]
    """Extract basic metadata from a DataFrame, including identifier detection."""
    meta = {}
    for col_name in df.columns:
        col = df[col_name]
        n_unique = col.nunique()

        # ── heuristics to detect identifiers ───────────────────────────
        is_id = (
            # very high cardinality …
            n_unique >= 0.9 * len(df)
            # … and either obvious name (*id, uuid, guid, …) …
            and any(tok in col_name.lower() for tok in ("id", "uuid", "guid"))
            # … or pandas says it’s non-numeric / not a date
            and not pd.api.types.is_numeric_dtype(col)
            and not pd.api.types.is_datetime64_any_dtype(col)
        )

        meta[col_name] = {
            "dtype": str(col.dtype),
            "n_unique": n_unique,
            "is_id": is_id,
        }
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "numeric_stats": df[numeric_cols].describe().to_dict() if numeric_cols else {},
        "categorical_unique_counts": {col: df[col].nunique() for col in cat_cols} if cat_cols else {},
        
        # Legacy fields preserved to prevent breaking old pipeline components immediately
        "summary": limited.describe(include="all").to_dict(),
        "metadata": meta,
    }


import re
from pathlib import Path
from typing import List
import yaml
from utils.security import secure_join

from .column_meta import ColumnMeta, apply_role_feedback

_ROLES_CACHE: dict | None = None


def _load_roles() -> dict:
    global _ROLES_CACHE
    if _ROLES_CACHE is None:
        base = Path(__file__).resolve().parents[1] / "config"
        roles_path = secure_join(base, "semantic_roles.yaml")
        with open(roles_path, "r", encoding="utf-8") as f:
            _ROLES_CACHE = yaml.safe_load(f) or {}
    return _ROLES_CACHE


from utils.role_mapper import map_description_to_role


def infer_column_meta(
    df: pd.DataFrame, descriptions: dict[str, str] | None = None
) -> List[ColumnMeta]:
    """Infer column roles for ``df`` using simple name heuristics.

    Optionally use ``descriptions`` to map free-text descriptions to roles.
    """
    roles = _load_roles()
    meta: List[ColumnMeta] = []
    descriptions = descriptions or {}
    for name in df.columns:
        best_role = "unknown"
        best_len = 0
        name_l = re.sub(r"[^a-z0-9]", "", name.lower())
        for role, phrases in roles.items():
            for phrase in phrases or []:
                phrase_l = re.sub(r"[^a-z0-9]", "", str(phrase).lower())
                if phrase_l and phrase_l in name_l and len(phrase_l) > best_len:
                    best_role = role
                    best_len = len(phrase_l)
        if name in descriptions and descriptions[name]:
            desc_role = map_description_to_role(descriptions[name])
            if desc_role != "unknown":
                best_role = desc_role
        confidence = 0.9 if best_role != "unknown" else 0.0
        meta.append(
            ColumnMeta(name=name, role=best_role, confidence=confidence)
        )
    meta = apply_role_feedback(meta, df)
    return meta


def merge_user_labels(meta: List[ColumnMeta], user_labels: dict[str, str]) -> List[ColumnMeta]:
    """Override roles in ``meta`` with ``user_labels``."""
    updated = []
    for m in meta:
        if m.name in user_labels:
            updated.append(
                ColumnMeta(
                    name=m.name,
                    role=user_labels[m.name],
                    confidence=1.0,
                    source="user",
                    description=m.description,
                )
            )
        else:
            updated.append(m)
    return updated
