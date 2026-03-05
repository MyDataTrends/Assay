"""
MCP Semantic Merge Tools.

Tools for intelligent dataset joining and enrichment using
semantic analysis of column names, types, and content.
"""
from __future__ import annotations  # Defer type annotation evaluation

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Lazy import - pandas is imported inside methods when needed
if TYPE_CHECKING:
    import pandas as pd

from . import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    register_category,
    success_response,
    error_response,
)

logger = logging.getLogger(__name__)


semantic_category = ToolCategory()
semantic_category.name = "semantic"
semantic_category.description = "Semantic dataset analysis and merge tools"


class SemanticAnalyzeDatasetsTool(BaseTool):
    """Analyze datasets for merge compatibility."""
    
    name = "semantic_analyze_datasets"
    description = (
        "Analyze multiple datasets to understand their structure, find common columns, "
        "and assess merge compatibility. Returns semantic similarity scores and "
        "suggested merge strategies."
    )
    category = "semantic"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "dataset_ids", "array",
                "List of dataset IDs to analyze",
                items={"type": "string"},
                required=True,
                examples=[["sales_2023", "sales_2024"]]
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        dataset_ids = arguments["dataset_ids"]
        
        if not session:
            return error_response("Session required")
        
        if len(dataset_ids) < 2:
            return error_response("At least 2 datasets required for comparison")
        
        datasets = {}
        for did in dataset_ids:
            df = session.get_dataset(did)
            if df is None:
                return error_response(f"Dataset not found: {did}")
            datasets[did] = df
        
        # Analyze each dataset
        analyses = {}
        for did, df in datasets.items():
            analyses[did] = {
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "numeric_cols": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                "text_cols": df.select_dtypes(include=['object']).columns.tolist(),
                "datetime_cols": df.select_dtypes(include=['datetime64']).columns.tolist(),
            }
        
        # Find common columns (by name)
        all_columns = [set(df.columns) for df in datasets.values()]
        common_columns = set.intersection(*all_columns) if all_columns else set()
        
        # Find similar columns (fuzzy matching on name)
        similar_columns = []
        checked_pairs = set()
        
        for did1, df1 in datasets.items():
            for did2, df2 in datasets.items():
                if did1 >= did2:
                    continue
                pair_key = (did1, did2)
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)
                
                for col1 in df1.columns:
                    for col2 in df2.columns:
                        if col1 == col2:
                            continue
                        # Simple similarity check
                        col1_lower = col1.lower().replace("_", "").replace("-", "")
                        col2_lower = col2.lower().replace("_", "").replace("-", "")
                        
                        if col1_lower == col2_lower or col1_lower in col2_lower or col2_lower in col1_lower:
                            similar_columns.append({
                                "dataset1": did1,
                                "column1": col1,
                                "dataset2": did2,
                                "column2": col2,
                                "reason": "name_similarity"
                            })
        
        # Suggest merge strategies
        suggestions = []
        if common_columns:
            suggestions.append({
                "strategy": "inner_join",
                "on": list(common_columns),
                "description": f"Join on {len(common_columns)} common columns"
            })
        
        if similar_columns:
            suggestions.append({
                "strategy": "mapped_join",
                "mappings": similar_columns[:5],
                "description": "Join using similar column mappings"
            })
        
        suggestions.append({
            "strategy": "concat",
            "description": "Concatenate datasets vertically (if same structure)"
        })
        
        return success_response({
            "datasets": analyses,
            "common_columns": list(common_columns),
            "similar_columns": similar_columns[:10],
            "suggested_strategies": suggestions,
        })


class SemanticSuggestMergeKeysTool(BaseTool):
    """Suggest optimal merge keys based on column semantics."""
    
    name = "semantic_suggest_merge_keys"
    description = (
        "Analyze two datasets and suggest the best columns to use as merge keys "
        "based on semantic analysis of column names, data types, and value distributions."
    )
    category = "semantic"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "source_dataset", "string",
                "Source dataset ID",
                required=True
            ),
            ToolParameter(
                "target_dataset", "string",
                "Target dataset ID to merge with",
                required=True
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        source_id = arguments["source_dataset"]
        target_id = arguments["target_dataset"]
        
        if not session:
            return error_response("Session required")
        
        source_df = session.get_dataset(source_id)
        target_df = session.get_dataset(target_id)
        
        if source_df is None:
            return error_response(f"Source dataset not found: {source_id}")
        if target_df is None:
            return error_response(f"Target dataset not found: {target_id}")
        
        suggestions = []
        
        # Check each column in source against target
        for src_col in source_df.columns:
            src_dtype = str(source_df[src_col].dtype)
            src_unique = source_df[src_col].nunique()
            src_unique_ratio = src_unique / len(source_df) if len(source_df) > 0 else 0
            
            for tgt_col in target_df.columns:
                tgt_dtype = str(target_df[tgt_col].dtype)
                
                score = 0
                reasons = []
                
                # Same name = high score
                if src_col.lower() == tgt_col.lower():
                    score += 50
                    reasons.append("exact_name_match")
                elif src_col.lower().replace("_", "") == tgt_col.lower().replace("_", ""):
                    score += 40
                    reasons.append("normalized_name_match")
                
                # Same dtype
                if src_dtype == tgt_dtype:
                    score += 20
                    reasons.append("same_dtype")
                
                # ID-like columns
                id_keywords = ["id", "key", "code", "number", "num", "no"]
                if any(kw in src_col.lower() for kw in id_keywords):
                    score += 15
                    reasons.append("id_column")
                
                # High uniqueness in source = likely a key
                if src_unique_ratio > 0.9:
                    score += 10
                    reasons.append("high_uniqueness")
                
                # Check for value overlap (sample)
                if score > 30 and src_dtype == tgt_dtype:
                    try:
                        src_values = set(source_df[src_col].dropna().head(1000).astype(str))
                        tgt_values = set(target_df[tgt_col].dropna().head(1000).astype(str))
                        overlap = len(src_values & tgt_values)
                        if overlap > 0:
                            overlap_ratio = overlap / min(len(src_values), len(tgt_values)) if src_values and tgt_values else 0
                            score += int(overlap_ratio * 30)
                            if overlap_ratio > 0.5:
                                reasons.append(f"value_overlap_{int(overlap_ratio*100)}%")
                    except Exception:
                        pass
                
                if score > 30:
                    suggestions.append({
                        "source_column": src_col,
                        "target_column": tgt_col,
                        "score": score,
                        "reasons": reasons,
                        "source_dtype": src_dtype,
                        "target_dtype": tgt_dtype,
                    })
        
        # Sort by score
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        
        return success_response({
            "suggestions": suggestions[:10],
            "best_match": suggestions[0] if suggestions else None,
            "source_columns": list(source_df.columns),
            "target_columns": list(target_df.columns),
        })


class SemanticMergeTool(BaseTool):
    """Execute semantic merge between datasets."""
    
    name = "semantic_merge"
    description = (
        "Merge two or more datasets using semantic key matching. "
        "Supports various merge strategies including inner, left, right, and outer joins."
    )
    category = "semantic"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "left_dataset", "string",
                "Left dataset ID",
                required=True,
                examples=["customers"]
            ),
            ToolParameter(
                "right_dataset", "string",
                "Right dataset ID",
                required=True,
                examples=["orders"]
            ),
            ToolParameter(
                "left_on", "array",
                "Columns from left dataset to merge on",
                items={"type": "string"},
                required=True,
                examples=[["customer_id"]]
            ),
            ToolParameter(
                "right_on", "array",
                "Columns from right dataset to merge on",
                items={"type": "string"},
                required=True,
                examples=[["user_id"]]
            ),
            ToolParameter(
                "how", "string",
                "Merge strategy",
                enum=["inner", "left", "right", "outer"],
                default="inner"
            ),
            ToolParameter(
                "save_as", "string",
                "Dataset ID to save merged result",
                required=True,
                examples=["customers_with_orders"]
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        left_id = arguments["left_dataset"]
        right_id = arguments["right_dataset"]
        left_on = arguments["left_on"]
        right_on = arguments["right_on"]
        how = arguments.get("how", "inner")
        save_as = arguments["save_as"]
        
        if not session:
            return error_response("Session required")
        
        left_df = session.get_dataset(left_id)
        right_df = session.get_dataset(right_id)
        
        if left_df is None:
            return error_response(f"Left dataset not found: {left_id}")
        if right_df is None:
            return error_response(f"Right dataset not found: {right_id}")
        
        # Validate columns
        for col in left_on:
            if col not in left_df.columns:
                return error_response(f"Column '{col}' not in left dataset")
        for col in right_on:
            if col not in right_df.columns:
                return error_response(f"Column '{col}' not in right dataset")
        
        if len(left_on) != len(right_on):
            return error_response("left_on and right_on must have same length")
        
        try:
            import pandas as pd
            # Perform merge
            merged_df = pd.merge(
                left_df,
                right_df,
                left_on=left_on,
                right_on=right_on,
                how=how,
                suffixes=('_left', '_right')
            )
            
            # Save to session
            session.add_dataset(save_as, merged_df, {
                "source": "semantic_merge",
                "left": left_id,
                "right": right_id,
                "merge_type": how,
            })
            
            return success_response({
                "dataset_id": save_as,
                "rows": len(merged_df),
                "columns": list(merged_df.columns),
                "left_rows": len(left_df),
                "right_rows": len(right_df),
                "merge_type": how,
                "match_rate": round(len(merged_df) / max(len(left_df), 1) * 100, 2),
            })
            
        except Exception as e:
            return error_response(f"Merge failed: {e}")


class SemanticScoreSimilarityTool(BaseTool):
    """Score semantic similarity between datasets."""
    
    name = "semantic_score_similarity"
    description = (
        "Calculate semantic similarity scores between datasets based on "
        "column structure, data types, and value distributions."
    )
    category = "semantic"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "dataset_ids", "array",
                "List of dataset IDs to compare",
                items={"type": "string"},
                required=True,
                examples=[["dataset_a", "dataset_b"]]
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        dataset_ids = arguments["dataset_ids"]
        
        if not session:
            return error_response("Session required")
        
        datasets = {}
        for did in dataset_ids:
            df = session.get_dataset(did)
            if df is None:
                return error_response(f"Dataset not found: {did}")
            datasets[did] = df
        
        # Calculate pairwise similarity
        similarities = []
        
        dataset_list = list(datasets.items())
        for i, (id1, df1) in enumerate(dataset_list):
            for id2, df2 in dataset_list[i+1:]:
                # Column overlap
                cols1 = set(df1.columns)
                cols2 = set(df2.columns)
                col_overlap = len(cols1 & cols2) / len(cols1 | cols2) if cols1 | cols2 else 0
                
                # dtype distribution similarity
                dtypes1 = df1.dtypes.value_counts(normalize=True).to_dict()
                dtypes2 = df2.dtypes.value_counts(normalize=True).to_dict()
                all_dtypes = set(dtypes1.keys()) | set(dtypes2.keys())
                dtype_sim = 1 - sum(
                    abs(dtypes1.get(dt, 0) - dtypes2.get(dt, 0)) 
                    for dt in all_dtypes
                ) / 2
                
                # Row count similarity
                row_ratio = min(len(df1), len(df2)) / max(len(df1), len(df2)) if max(len(df1), len(df2)) > 0 else 1
                
                # Overall score
                overall = (col_overlap * 0.5 + dtype_sim * 0.3 + row_ratio * 0.2)
                
                similarities.append({
                    "dataset1": id1,
                    "dataset2": id2,
                    "column_overlap": round(col_overlap, 3),
                    "dtype_similarity": round(dtype_sim, 3),
                    "row_ratio": round(row_ratio, 3),
                    "overall_score": round(overall, 3),
                    "common_columns": list(cols1 & cols2),
                })
        
        # Sort by overall score
        similarities.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return success_response({
            "similarities": similarities,
            "most_similar": similarities[0] if similarities else None,
        })


class SemanticEnrichTool(BaseTool):
    """Enrich dataset with public data lake sources."""
    
    name = "semantic_enrich"
    description = (
        "Enrich a dataset by finding and merging relevant public datasets "
        "from the data lake. Uses semantic matching to find compatible sources."
    )
    category = "semantic"
    requires_session = True
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "dataset_id", "string",
                "Dataset ID to enrich",
                required=True,
                examples=["sales_2024"]
            ),
            ToolParameter(
                "category", "string",
                "Category hint for finding relevant data (e.g., 'demographic', 'geographic')",
                examples=["economic"]
            ),
            ToolParameter(
                "datalake_path", "string",
                "Path to scan for public datasets",
                default="datasets"
            ),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        dataset_id = arguments["dataset_id"]
        category = arguments.get("category")
        datalake_path = arguments.get("datalake_path", "datasets")
        
        if not session:
            return error_response("Session required")
        
        df = session.get_dataset(dataset_id)
        if df is None:
            return error_response(f"Dataset not found: {dataset_id}")
        
        # Try to use existing Assay enrichment
        try:
            from orchestration.semantic_enricher import SemanticEnricher
            
            enricher = SemanticEnricher()
            enriched_df = enricher.enrich(
                df,
                category=category,
            )
            
            enriched_id = f"{dataset_id}_enriched"
            session.add_dataset(enriched_id, enriched_df, {
                "source": "semantic_enrich",
                "original": dataset_id,
                "category": category,
            })
            
            new_columns = [c for c in enriched_df.columns if c not in df.columns]
            
            return success_response({
                "dataset_id": enriched_id,
                "original_columns": list(df.columns),
                "new_columns": new_columns,
                "rows": len(enriched_df),
            })
            
        except ImportError:
            return error_response("SemanticEnricher not available")
        except Exception as e:
            return error_response(f"Enrichment failed: {e}")


# Register all tools
semantic_category.register(SemanticAnalyzeDatasetsTool())
semantic_category.register(SemanticSuggestMergeKeysTool())
semantic_category.register(SemanticMergeTool())
semantic_category.register(SemanticScoreSimilarityTool())
semantic_category.register(SemanticEnrichTool())

register_category(semantic_category)
