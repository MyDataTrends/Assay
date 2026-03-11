"""
MCP Decision Tools - Analysis routing and intent classification.
"""

import logging
from typing import Any, Dict, List
from . import BaseTool, ToolCategory, ToolParameter, register_category, success_response, error_response

logger = logging.getLogger(__name__)
decision_category = ToolCategory()
decision_category.name = "decision"
decision_category.description = "Analysis routing and decision-making tools"


class DecideAnalysisTypeTool(BaseTool):
    name = "decide_analysis_type"
    description = "Determine appropriate analysis type (visualization, modeling, exploration) for a query."
    category = "decision"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("query", "string", "Natural language query", required=True, examples=["show me a chart of sales", "predict next month revenue"]),
            ToolParameter("dataset_id", "string", "Dataset ID for context", examples=["sales_2024"]),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        query = arguments["query"].lower()
        
        if any(w in query for w in ["chart", "plot", "show", "visualize"]):
            action = "visualization"
        elif any(w in query for w in ["predict", "forecast", "classify", "model"]):
            action = "modeling"
        else:
            action = "analysis"
        
        try:
            from orchestration.cascade_planner import classify_intent
            intent, confidence = classify_intent(arguments["query"])
            action_map = {
                "MODEL_TRAIN": "modeling", "MODEL_PREDICT": "modeling",
                "VISUALIZE": "visualization", "DESCRIBE_DATA": "exploration",
            }
            mapped_action = action_map.get(intent.name, "analysis") if hasattr(intent, "name") else "analysis"
            return success_response({"action": mapped_action, "params": {}, "confidence": confidence})
        except Exception:
            return success_response({"action": action, "params": {}, "confidence": 0.5})


class ClassifyIntentTool(BaseTool):
    name = "classify_intent"
    description = "Parse user intent into structured action with type, columns, and parameters."
    category = "decision"
    requires_session = False
    
    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter("query", "string", "Query to parse", required=True, examples=["compare revenue by region", "find outliers in temperature"])]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        try:
            from orchestration.cascade_planner import classify_intent
            intent, confidence = classify_intent(arguments["query"])
            if intent:
                return success_response({
                    "intent": intent.value if hasattr(intent, "value") else str(intent), 
                    "confidence": confidence,
                    "params": {} # Cascade planner doesn't return params here, those are filled during planning
                })
        except Exception:
            pass
        return success_response({"intent": "unknown", "params": {}})


class RouteAnalysisTool(BaseTool):
    name = "route_analysis"
    description = "Route to appropriate analyzer based on data characteristics."
    category = "decision"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("dataset_id", "string", "Dataset ID", required=True, examples=["fred_gdp"]),
            ToolParameter("hints", "object", "Routing hints"),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        numeric = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        datetime = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if len(datetime) > 0:
            route = "time_series"
        elif len(numeric) >= 2:
            route = "regression"
        else:
            route = "exploratory"
        
        return success_response({"route": route, "numeric_cols": numeric, "datetime_cols": datetime})


class AssessModelabilityTool(BaseTool):
    name = "assess_modelability"
    description = "Check if data is suitable for ML modeling."
    category = "decision"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("dataset_id", "string", "Dataset ID", required=True, examples=["sales_2024"]),
            ToolParameter("target_column", "string", "Target column", examples=["revenue", "churn"]),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        issues, score = [], 100
        if len(df) < 10:
            issues.append("Too few rows"); score -= 50
        if df.isna().mean().mean() > 0.5:
            issues.append("Too much missing data"); score -= 30
        
        target = arguments.get("target_column")
        if target and target in df.columns and df[target].nunique() < 2:
            issues.append("No target variance"); score -= 40
        
        return success_response({"modelable": score >= 50, "score": max(0, score), "issues": issues})


class SuggestTargetVariableTool(BaseTool):
    name = "suggest_target_variable"
    description = "Recommend target variable for prediction."
    category = "decision"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("dataset_id", "string", "Dataset ID", required=True, examples=["sales_2024"]),
            ToolParameter("context", "string", "What to predict", examples=["next quarter revenue"]),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        suggestions = []
        for col in df.columns:
            score = 0
            if any(kw in col.lower() for kw in ["target", "label", "outcome", "y"]):
                score += 30
            if df[col].dtype in ['int64', 'float64'] and df[col].std() > 0:
                score += 10
            if score > 0:
                suggestions.append({"column": col, "score": score, "dtype": str(df[col].dtype)})
        
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return success_response({"suggestions": suggestions[:5]})


decision_category.register(DecideAnalysisTypeTool())
decision_category.register(ClassifyIntentTool())
decision_category.register(RouteAnalysisTool())
decision_category.register(AssessModelabilityTool())
decision_category.register(SuggestTargetVariableTool())
register_category(decision_category)
