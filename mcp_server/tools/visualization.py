"""
MCP Visualization Tools - Chart generation and recommendations.
"""

import logging
import uuid
import base64
from io import BytesIO
from typing import Any, Dict, List
from . import BaseTool, ToolCategory, ToolParameter, register_category, success_response, error_response

logger = logging.getLogger(__name__)
viz_category = ToolCategory()
viz_category.name = "visualization"
viz_category.description = "Chart generation and visualization tools"


def _export_fig(fig, export_format: str, chart_id: str, extra: dict = None):
    """Shared helper to export a Plotly figure in the requested format."""
    result = {"chart_id": chart_id}
    if extra:
        result.update(extra)

    fmt = (export_format or "plotly_json").lower()
    if fmt == "plotly_json":
        result["plotly_json"] = fig.to_json()
    elif fmt in ("png_base64", "svg"):
        img_fmt = "png" if fmt == "png_base64" else "svg"
        try:
            img_bytes = fig.to_image(format=img_fmt)
            if img_fmt == "png":
                result["png_base64"] = base64.b64encode(img_bytes).decode()
            else:
                result["svg"] = img_bytes.decode("utf-8")
        except Exception as e:
            err = str(e)
            if "kaleido" in err.lower() or "orca" in err.lower():
                return error_response(
                    f"Image export requires kaleido: pip install kaleido.  "
                    f"Falling back to plotly_json."
                )
            return error_response(f"Image export failed: {e}")
    else:
        result["plotly_json"] = fig.to_json()

    return success_response(result)


class SuggestVisualizationsTool(BaseTool):
    name = "suggest_visualizations"
    description = "Get AI-recommended visualizations for a dataset based on its structure."
    category = "visualization"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("dataset_id", "string", "Dataset ID", required=True, examples=["fred_gdp", "sales_2024"]),
            ToolParameter("question", "string", "Optional question context", examples=["what drives revenue?"]),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        suggestions = []
        numeric = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if len(numeric) >= 2:
            suggestions.append({"type": "scatter", "x": numeric[0], "y": numeric[1], 
                              "reason": "Compare two numeric variables"})
            suggestions.append({"type": "heatmap", "columns": numeric[:5],
                              "reason": "Show correlations between numeric columns"})
        if numeric:
            suggestions.append({"type": "histogram", "column": numeric[0],
                              "reason": "Show distribution"})
        if categorical and numeric:
            suggestions.append({"type": "bar", "x": categorical[0], "y": numeric[0],
                              "reason": "Compare categories"})
        if datetime and numeric:
            suggestions.append({"type": "line", "x": datetime[0], "y": numeric[0],
                              "reason": "Time series trend"})
        if categorical:
            suggestions.append({"type": "pie", "column": categorical[0],
                              "reason": "Show proportions"})
        
        # Try LLM suggestions
        try:
            from llm_manager.llm_interface import suggest_visualizations, is_llm_available
            if is_llm_available():
                llm_suggestions = suggest_visualizations(df)
                if llm_suggestions:
                    suggestions = llm_suggestions + suggestions
        except Exception:
            pass
        
        return success_response({"suggestions": suggestions[:6]})


class CreateChartTool(BaseTool):
    name = "create_chart"
    description = "Generate a specific chart type from dataset."
    category = "visualization"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("dataset_id", "string", "Dataset ID", required=True, examples=["fred_gdp", "sales_2024"]),
            ToolParameter("chart_type", "string", "Chart type", required=True,
                         enum=["bar", "line", "scatter", "histogram", "pie", "box", "heatmap"]),
            ToolParameter("x", "string", "X-axis column", examples=["date", "category"]),
            ToolParameter("y", "string", "Y-axis column", examples=["value", "revenue"]),
            ToolParameter("color", "string", "Color by column", examples=["region"]),
            ToolParameter("title", "string", "Chart title", examples=["Sales by Region"]),
            ToolParameter("export_format", "string", "Output format",
                         enum=["plotly_json", "png_base64", "svg"],
                         default="plotly_json"),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError:
            return error_response("Plotly required: pip install plotly")
        
        chart_type = arguments["chart_type"]
        x, y = arguments.get("x"), arguments.get("y")
        color = arguments.get("color")
        title = arguments.get("title", f"{chart_type.title()} Chart")
        
        try:
            if chart_type == "bar":
                fig = px.bar(df, x=x, y=y, color=color, title=title)
            elif chart_type == "line":
                fig = px.line(df, x=x, y=y, color=color, title=title)
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x, y=y, color=color, title=title)
            elif chart_type == "histogram":
                fig = px.histogram(df, x=x or y, title=title)
            elif chart_type == "pie":
                fig = px.pie(df, names=x, title=title)
            elif chart_type == "box":
                fig = px.box(df, x=x, y=y, title=title)
            elif chart_type == "heatmap":
                corr = df.select_dtypes(include=['int64', 'float64']).corr()
                fig = px.imshow(corr, text_auto=".2f", title=title)
            else:
                return error_response(f"Unknown chart type: {chart_type}")
            
            fig.update_layout(template="plotly_dark")
            chart_id = f"chart_{uuid.uuid4().hex[:8]}"
            
            # Store chart
            session.add_chart(chart_id, fig.to_dict(), {
                "type": chart_type, "x": x, "y": y, "title": title
            })
            
            return _export_fig(fig, arguments.get("export_format", "plotly_json"),
                               chart_id, {"type": chart_type, "title": title})
        except Exception as e:
            return error_response(f"Chart creation failed: {e}")


class CreateCorrelationHeatmapTool(BaseTool):
    name = "create_correlation_heatmap"
    description = "Generate correlation matrix heatmap for numeric columns."
    category = "visualization"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("dataset_id", "string", "Dataset ID", required=True, examples=["sales_2024"]),
            ToolParameter("columns", "array", "Columns to include", items={"type": "string"}, examples=[["price", "quantity", "revenue"]]),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        columns = arguments.get("columns")
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        if columns:
            numeric_df = numeric_df[[c for c in columns if c in numeric_df.columns]]
        
        if len(numeric_df.columns) < 2:
            return error_response("Need at least 2 numeric columns")
        
        try:
            import plotly.express as px
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=".2f", title="Correlation Matrix",
                          color_continuous_scale="RdBu_r")
            fig.update_layout(template="plotly_dark")
            
            chart_id = f"corr_{uuid.uuid4().hex[:8]}"
            session.add_chart(chart_id, fig.to_dict(), {"type": "correlation"})
            
            # Find strong correlations
            strong = []
            for i, c1 in enumerate(corr.columns):
                for c2 in corr.columns[i+1:]:
                    if abs(corr.loc[c1, c2]) > 0.5:
                        strong.append({"col1": c1, "col2": c2, "corr": round(corr.loc[c1, c2], 3)})
            
            return success_response({
                "chart_id": chart_id,
                "strong_correlations": strong,
                "plotly_json": fig.to_json(),
            })
        except ImportError:
            return error_response("Plotly required")


class CreateDistributionPlotTool(BaseTool):
    name = "create_distribution_plot"
    description = "Generate histogram/distribution plot for a column."
    category = "visualization"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("dataset_id", "string", "Dataset ID", required=True, examples=["fred_gdp"]),
            ToolParameter("column", "string", "Column to plot", required=True, examples=["value", "price"]),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        col = arguments["column"]
        if col not in df.columns:
            return error_response(f"Column not found: {col}")
        
        try:
            import plotly.express as px
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            fig.update_layout(template="plotly_dark")
            
            chart_id = f"dist_{uuid.uuid4().hex[:8]}"
            session.add_chart(chart_id, fig.to_dict(), {"type": "distribution", "column": col})
            
            # Stats
            stats = {}
            if df[col].dtype in ['int64', 'float64']:
                stats = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }
            
            return success_response({"chart_id": chart_id, "statistics": stats, "plotly_json": fig.to_json()})
        except ImportError:
            return error_response("Plotly required")


class CreateTimeSeriesPlotTool(BaseTool):
    name = "create_time_series_plot"
    description = "Generate time series line plot."
    category = "visualization"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("dataset_id", "string", "Dataset ID", required=True, examples=["fred_gdp"]),
            ToolParameter("time_col", "string", "Time/date column name", required=True, examples=["date", "observation_date", "timestamp"]),
            ToolParameter("value_col", "string", "Value column to plot", required=True, examples=["value", "price", "amount"]),
            ToolParameter("export_format", "string", "Output format",
                         enum=["plotly_json", "png_base64", "svg"],
                         default="plotly_json"),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        df = session.get_dataset(arguments["dataset_id"])
        if df is None:
            return error_response("Dataset not found")
        
        time_col, value_col = arguments["time_col"], arguments["value_col"]
        
        try:
            import plotly.express as px
            df_sorted = df.sort_values(time_col)
            fig = px.line(df_sorted, x=time_col, y=value_col, title=f"{value_col} Over Time")
            fig.update_layout(template="plotly_dark")
            
            chart_id = f"ts_{uuid.uuid4().hex[:8]}"
            session.add_chart(chart_id, fig.to_dict(), {"type": "time_series"})
            
            return _export_fig(fig, arguments.get("export_format", "plotly_json"), chart_id)
        except ImportError:
            return error_response("Plotly required")


class ExportChartTool(BaseTool):
    name = "export_chart"
    description = "Export a generated chart to various formats."
    category = "visualization"
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("chart_id", "string", "Chart ID", required=True),
            ToolParameter("format", "string", "Export format", enum=["png", "html", "json"], default="json"),
        ]
    
    async def execute(self, arguments: Dict[str, Any], session=None) -> Dict[str, Any]:
        if not session:
            return error_response("Session required")
        
        chart_id = arguments["chart_id"]
        chart_data = session.charts.get(chart_id)
        if not chart_data:
            return error_response(f"Chart not found: {chart_id}")
        
        export_format = arguments.get("format", "json")
        
        try:
            import plotly.graph_objects as go
            fig = go.Figure(chart_data["data"])
            
            if export_format == "json":
                return success_response({"format": "json", "data": fig.to_json()})
            elif export_format == "html":
                return success_response({"format": "html", "data": fig.to_html()})
            elif export_format == "png":
                img_bytes = fig.to_image(format="png")
                return success_response({"format": "png", "data": base64.b64encode(img_bytes).decode()})
        except Exception as e:
            return error_response(f"Export failed: {e}")


viz_category.register(SuggestVisualizationsTool())
viz_category.register(CreateChartTool())
viz_category.register(CreateCorrelationHeatmapTool())
viz_category.register(CreateDistributionPlotTool())
viz_category.register(CreateTimeSeriesPlotTool())
viz_category.register(ExportChartTool())
register_category(viz_category)
