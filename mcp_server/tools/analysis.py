"""
MCP Analysis Tools - Statistical and ML operations.
"""
from __future__ import annotations  # Defer type annotation evaluation

import logging
import uuid
from typing import Any, Dict, List, TYPE_CHECKING

# Lazy imports - heavy libraries imported inside methods when needed
if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
from . import BaseTool, ToolCategory, ToolParameter, register_category, success_response, error_response

logger = logging.getLogger(__name__)
analysis_category = ToolCategory()
analysis_category.name = "analysis"
analysis_category.description = "Statistical and machine learning analysis tools"


class DescribeDatasetTool(BaseTool):
    name = "describe_dataset"
    description = "Get comprehensive statistics for a dataset."
    category = "analysis"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter("dataset_id", "string", "Dataset ID", required=True, examples=["fred_gdp", "sales_2024"])]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        import pandas as pd
        import numpy as np
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        result = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": [],
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "missing_total": int(df.isna().sum().sum()),
        }
        
        for col in df.columns:
            info = {"name": col, "dtype": str(df[col].dtype), "missing": int(df[col].isna().sum())}
            if df[col].dtype in ['int64', 'float64']:
                info["stats"] = {
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                }
            else:
                info["unique"] = int(df[col].nunique())
                info["top_values"] = {str(k): int(v) for k, v in df[col].value_counts().head(3).items()}
            result["columns"].append(info)
        
        return success_response(result)


class DetectAnomaliesTool(BaseTool):
    name = "detect_anomalies"
    description = "Find anomalies/outliers in dataset columns."
    category = "analysis"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("dataset_id", "string", "Dataset ID", required=True),
            ToolParameter("columns", "array", "Columns to check", items={"type": "string"}),
            ToolParameter("method", "string", "Detection method", enum=["iqr", "zscore"], default="iqr"),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        import pandas as pd
        import numpy as np
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        columns = arguments.get("columns") or df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        method = arguments.get("method", "iqr")
        
        anomalies = {}
        for col in columns:
            if col not in df.columns or df[col].dtype not in ['int64', 'float64']:
                continue
            
            data = df[col].dropna()
            if method == "iqr":
                q1, q3 = data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = ((data < lower) | (data > upper)).sum()
            else:
                z = np.abs((data - data.mean()) / data.std())
                outliers = (z > 3).sum()
            
            if outliers > 0:
                anomalies[col] = {"count": int(outliers), "pct": round(outliers / len(data) * 100, 2)}
        
        return success_response({"anomalies": anomalies, "method": method})


class RunCorrelationAnalysisTool(BaseTool):
    name = "run_correlation_analysis"
    description = "Compute correlations between numeric columns."
    category = "analysis"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter("dataset_id", "string", "Dataset ID", required=True)]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        import pandas as pd
        import numpy as np
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        numeric = df.select_dtypes(include=['int64', 'float64'])
        if len(numeric.columns) < 2:
            return error_response("Need at least 2 numeric columns")
        
        corr = numeric.corr()
        
        pairs = []
        for i, c1 in enumerate(corr.columns):
            for c2 in corr.columns[i+1:]:
                pairs.append({"col1": c1, "col2": c2, "correlation": round(corr.loc[c1, c2], 4)})
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return success_response({
            "matrix": corr.to_dict(),
            "top_correlations": pairs[:10],
        })


class ClusterDataTool(BaseTool):
    name = "cluster_data"
    description = "Run clustering algorithm on dataset."
    category = "analysis"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("dataset_id", "string", "Dataset ID", required=True),
            ToolParameter("n_clusters", "number", "Number of clusters", default=3),
            ToolParameter("columns", "array", "Columns to use", items={"type": "string"}),
            ToolParameter("save_as", "string", "Save clustered dataset as"),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        import pandas as pd
        import numpy as np
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return error_response("sklearn required")
        
        columns = arguments.get("columns") or df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        n_clusters = arguments.get("n_clusters", 3)
        
        X = df[columns].dropna()
        if len(X) < n_clusters:
            return error_response("Not enough data points")
        
        X_scaled = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        result_df = df.loc[X.index].copy()
        result_df["cluster"] = labels
        
        if arguments.get("save_as"):
            session.add_dataset(arguments["save_as"], result_df, {"source": "clustering"})
        
        cluster_sizes = pd.Series(labels).value_counts().to_dict()
        return success_response({
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
            "dataset_id": arguments.get("save_as"),
        })


class ForecastTool(BaseTool):
    name = "forecast"
    description = "Run time series forecasting."
    category = "analysis"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("dataset_id", "string", "Dataset ID", required=True),
            ToolParameter("target", "string", "Target column", required=True),
            ToolParameter("periods", "number", "Periods to forecast", default=10),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        import pandas as pd
        import numpy as np
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        target = arguments["target"]
        periods = arguments.get("periods", 10)
        
        if target not in df.columns:
            return error_response(f"Column not found: {target}")
        
        # Simple moving average forecast
        values = df[target].dropna().values
        if len(values) < 3:
            return error_response("Not enough data")
        
        window = min(5, len(values) // 2)
        ma = np.convolve(values, np.ones(window)/window, mode='valid')
        trend = (ma[-1] - ma[0]) / len(ma) if len(ma) > 1 else 0
        
        forecast = [float(values[-1] + trend * (i + 1)) for i in range(periods)]
        
        return success_response({
            "forecast": forecast,
            "periods": periods,
            "method": "moving_average_trend",
        })


class RunClassificationTool(BaseTool):
    name = "run_classification"
    description = "Train a classification model."
    category = "analysis"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("dataset_id", "string", "Dataset ID", required=True),
            ToolParameter("target", "string", "Target column", required=True),
            ToolParameter("features", "array", "Feature columns", items={"type": "string"}),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        import pandas as pd
        import numpy as np
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
        except ImportError:
            return error_response("sklearn required")
        
        target = arguments["target"]
        features = arguments.get("features") or [c for c in df.select_dtypes(include=['int64', 'float64']).columns if c != target]
        
        if target not in df.columns:
            return error_response(f"Target not found: {target}")
        
        df_clean = df[features + [target]].dropna()
        X, y = df_clean[features], df_clean[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        model_id = f"clf_{uuid.uuid4().hex[:8]}"
        session.add_model(model_id, model, {"type": "classification", "target": target, "features": features})
        
        return success_response({
            "model_id": model_id,
            "accuracy": round(accuracy, 4),
            "feature_importance": dict(zip(features, [round(f, 4) for f in model.feature_importances_])),
        })


class RunRegressionTool(BaseTool):
    name = "run_regression"
    description = "Train a regression model."
    category = "analysis"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("dataset_id", "string", "Dataset ID", required=True),
            ToolParameter("target", "string", "Target column", required=True),
            ToolParameter("features", "array", "Feature columns", items={"type": "string"}),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        import pandas as pd
        import numpy as np
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_absolute_error
        except ImportError:
            return error_response("sklearn required")
        
        target = arguments["target"]
        features = arguments.get("features") or [c for c in df.select_dtypes(include=['int64', 'float64']).columns if c != target]
        
        df_clean = df[features + [target]].dropna()
        X, y = df_clean[features], df_clean[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        model_id = f"reg_{uuid.uuid4().hex[:8]}"
        session.add_model(model_id, model, {"type": "regression", "target": target, "features": features})
        
        return success_response({
            "model_id": model_id,
            "r2_score": round(r2, 4),
            "mae": round(mae, 4),
            "feature_importance": dict(zip(features, [round(f, 4) for f in model.feature_importances_])),
        })


class ExplainPredictionsTool(BaseTool):
    name = "explain_predictions"
    description = "Get feature importance and model explanations."
    category = "analysis"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [ToolParameter("model_id", "string", "Model ID", required=True)]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        import pandas as pd
        import numpy as np
        if not session:
            return error_response("Session required")
        
        model_id = arguments["model_id"]
        model_data = session.models.get(model_id)
        if not model_data:
            return error_response(f"Model not found: {model_id}")
        
        model = model_data["model"]
        metadata = model_data["metadata"]
        
        if hasattr(model, "feature_importances_"):
            features = metadata.get("features", [])
            importance = dict(zip(features, [round(f, 4) for f in model.feature_importances_]))
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            return success_response({
                "feature_importance": dict(sorted_imp),
                "top_features": [f[0] for f in sorted_imp[:5]],
                "model_type": metadata.get("type"),
            })
        
        return success_response({"message": "Feature importance not available for this model"})


analysis_category.register(DescribeDatasetTool())
analysis_category.register(DetectAnomaliesTool())
analysis_category.register(RunCorrelationAnalysisTool())
analysis_category.register(ClusterDataTool())
analysis_category.register(ForecastTool())
analysis_category.register(RunClassificationTool())
analysis_category.register(RunRegressionTool())
analysis_category.register(ExplainPredictionsTool())
register_category(analysis_category)
