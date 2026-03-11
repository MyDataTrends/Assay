from dataclasses import dataclass

@dataclass
class ColumnMeta:
    """Metadata describing a single DataFrame column."""

    name: str
    role: str
    confidence: float
    source: str = "heuristic"
    description: str | None = None

import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
from utils.security import secure_join



FEEDBACK_PATH = Path("feedback/role_corrections.json")


def _load_role_corrections(df: pd.DataFrame, file_path: Path = FEEDBACK_PATH) -> dict[str, str]:
    dest_dir = file_path.parent
    safe_path = secure_join(dest_dir, file_path.name)
    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
        
    from .save_meta import _hash_df
    return data.get(_hash_df(df), {})


def apply_role_feedback(
    meta: List[ColumnMeta],
    df: pd.DataFrame,
    file_path: Path = FEEDBACK_PATH,
) -> List[ColumnMeta]:
    """Return ``meta`` updated with stored feedback for ``df`` if present."""
    corrections = _load_role_corrections(df, file_path)
    if not corrections:
        return meta
    logging.info("Applying role corrections from feedback")
    updated = []
    for m in meta:
        if m.name in corrections:
            updated.append(
                ColumnMeta(
                    name=m.name,
                    role=corrections[m.name],
                    confidence=1.0,
                    source="feedback",
                    description=m.description,
                )
            )
        else:
            updated.append(m)
    return updated
