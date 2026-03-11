"""
Tool Registry for Cascade Planner.

Provides structured, safe tool definitions replacing raw exec():
- Multi-level granularity (fine-grained + coarse operations)
- Schema validation for inputs/outputs
- Built-in error handling and retry logic
- Integration with LLM learning system

Usage:
    from orchestration.tool_registry import get_tool, invoke_tool, TOOL_CATEGORIES
    
    tool = get_tool("data_profiler")
    result = invoke_tool("data_profiler", {"df": df})
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categories for hierarchical organization."""
    DATA_INSPECTION = "data_inspection"
    DATA_TRANSFORM = "data_transform"
    VISUALIZATION = "visualization"
    MODELING = "modeling"
    EXPORT = "export"
    EXTERNAL = "external"


class ToolGranularity(Enum):
    """Tool granularity levels."""
    FINE = "fine"      # Single operation (filter_rows, group_by)
    COARSE = "coarse"  # Multi-step composite (transform_data, analyze_data)


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": str(self.output)[:500] if self.output else None,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class Tool:
    """A registered tool with metadata and handler."""
    name: str
    description: str
    category: ToolCategory
    granularity: ToolGranularity
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    handler: Callable[..., Any]
    requires_llm: bool = False
    max_retries: int = 3
    fallback_tool: Optional[str] = None
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> tuple[bool, str]:
        """Validate inputs against schema."""
        for key, spec in self.input_schema.items():
            required = spec.get("required", False)
            expected_type = spec.get("type")
            
            if required and key not in inputs:
                return False, f"Missing required input: {key}"
            
            if key in inputs and expected_type:
                value = inputs[key]
                if expected_type == "dataframe" and not isinstance(value, pd.DataFrame):
                    return False, f"Input '{key}' must be a DataFrame"
                elif expected_type == "str" and not isinstance(value, str):
                    return False, f"Input '{key}' must be a string"
                elif expected_type == "list" and not isinstance(value, list):
                    return False, f"Input '{key}' must be a list"
        
        return True, ""


# =============================================================================
# Tool Handlers
# =============================================================================

def _handle_data_profiler(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Generate comprehensive data profile."""
    profile = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }
    
    # Numeric stats
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        profile["numeric_stats"] = df[numeric_cols].describe().to_dict()
    
    # Categorical stats
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        profile["categorical_unique_counts"] = {col: df[col].nunique() for col in cat_cols}
    
    return profile


def _handle_basic_stats(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Basic statistics fallback."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "describe": df.describe().to_dict() if len(df.select_dtypes(include=["number"]).columns) > 0 else {},
    }


def _handle_filter_rows(df: pd.DataFrame, column: str, operator: str, value: Any, **kwargs) -> pd.DataFrame:
    """Filter rows based on condition."""
    # Robustness: Try to convert column to numeric if value is numeric
    if isinstance(value, (int, float)):
        # Check if column is object/string but should be numeric
        if df[column].dtype == 'object' or str(df[column].dtype) == 'string':
            try:
                # Attempt coercion
                temp_col = pd.to_numeric(df[column], errors='coerce')
                # If we successfully converted some values (not all NaNs), use it
                if not temp_col.isna().all():
                     df = df.copy()
                     df[column] = temp_col
            except (ValueError, TypeError):
                pass 

    if operator == "==":
        return df[df[column] == value]
    elif operator == "!=":
        return df[df[column] != value]
    elif operator == ">":
        return df[df[column] > value]
    elif operator == "<":
        return df[df[column] < value]
    elif operator == ">=":
        return df[df[column] >= value]
    elif operator == "<=":
        return df[df[column] <= value]
    elif operator == "contains":
        return df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
    elif operator == "isin":
        return df[df[column].isin(value)]
    else:
        raise ValueError(f"Unknown operator: {operator}")


def _handle_group_by(df: pd.DataFrame, group_cols: List[str], agg_dict: Dict[str, str], **kwargs) -> pd.DataFrame:
    """Group and aggregate data."""
    return df.groupby(group_cols).agg(agg_dict).reset_index()


def _handle_select_columns(df: pd.DataFrame, columns: List[str], **kwargs) -> pd.DataFrame:
    """Select specific columns."""
    return df[columns]


def _handle_sort_data(df: pd.DataFrame, by: List[str], ascending: bool = True, **kwargs) -> pd.DataFrame:
    """Sort DataFrame."""
    return df.sort_values(by=by, ascending=ascending)


def _handle_fill_missing(df: pd.DataFrame, strategy: str = "mean", columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    """Fill missing values."""
    df = df.copy()
    cols = columns or df.select_dtypes(include=["number"]).columns.tolist()
    
    for col in cols:
        if col not in df.columns:
            continue
        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0)
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "drop":
            df = df.dropna(subset=[col])
    
    return df


def _handle_pandas_transform(df: pd.DataFrame, operations: List[Dict[str, Any]], **kwargs) -> pd.DataFrame:
    """Execute a sequence of pandas operations (coarse tool)."""
    result = df.copy()
    
    for op in operations:
        op_type = op.get("operation")
        params = op.get("params", {})
        
        if op_type == "filter":
            result = _handle_filter_rows(result, **params)
        elif op_type == "group_by":
            result = _handle_group_by(result, **params)
        elif op_type == "select":
            result = _handle_select_columns(result, **params)
        elif op_type == "sort":
            result = _handle_sort_data(result, **params)
        elif op_type == "fill_missing":
            result = _handle_fill_missing(result, **params)
        else:
            logger.warning(f"Unknown operation: {op_type}")
    
    return result


def _handle_chart_generator(df: pd.DataFrame, chart_type: str, x: str, y: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Generate chart specification."""
    try:
        import plotly.express as px
        
        if chart_type == "bar":
            fig = px.bar(df, x=x, y=y)
        elif chart_type == "line":
            fig = px.line(df, x=x, y=y)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x)
        elif chart_type == "pie":
            fig = px.pie(df, names=x, values=y)
        elif chart_type == "heatmap":
            # Correlation heatmap
            import plotly.graph_objects as go
            numeric_df = df.select_dtypes(include=["number"])
            corr = numeric_df.corr()
            fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns))
        else:
            fig = px.bar(df, x=x, y=y)  # Default to bar
        
        return {"figure": fig, "chart_type": chart_type}
    
    except ImportError:
        return {"error": "plotly not available", "chart_type": chart_type}


def _handle_table_display(df: pd.DataFrame, max_rows: int = 100, **kwargs) -> Dict[str, Any]:
    """Fallback table display."""
    return {
        "data": df.head(max_rows).to_dict(orient="records"),
        "total_rows": len(df),
        "columns": list(df.columns),
    }

def _handle_custom_python(df: pd.DataFrame, query: str, context: str = "", **kwargs) -> Dict[str, Any]:
    """Execute dynamically generated python code for complex queries."""
    from ui.chat_logic import execute_analysis_with_retry
    success, result, code, error = execute_analysis_with_retry(df, query, context=context)
    if success:
        return {"result": result, "code_generated": code}
    else:
        # Return error message gracefully rather than throwing exception to allow Cascade to handle it
        return {"error": error, "code_generated": code}


# =============================================================================
# Tool Registry
# =============================================================================

TOOL_REGISTRY: Dict[str, Tool] = {
    # Data Inspection - Fine
    "data_profiler": Tool(
        name="data_profiler",
        description="Generate comprehensive data profile with statistics",
        category=ToolCategory.DATA_INSPECTION,
        granularity=ToolGranularity.COARSE,
        input_schema={"df": {"type": "dataframe", "required": True}},
        output_schema={"profile": {"type": "dict"}},
        handler=_handle_data_profiler,
        fallback_tool="basic_stats",
    ),
    "basic_stats": Tool(
        name="basic_stats",
        description="Basic DataFrame statistics (fallback)",
        category=ToolCategory.DATA_INSPECTION,
        granularity=ToolGranularity.FINE,
        input_schema={"df": {"type": "dataframe", "required": True}},
        output_schema={"stats": {"type": "dict"}},
        handler=_handle_basic_stats,
    ),
    
    # Data Transform - Fine
    "filter_rows": Tool(
        name="filter_rows",
        description="Filter DataFrame rows based on condition",
        category=ToolCategory.DATA_TRANSFORM,
        granularity=ToolGranularity.FINE,
        input_schema={
            "df": {"type": "dataframe", "required": True},
            "column": {"type": "str", "required": True},
            "operator": {"type": "str", "required": True},
            "value": {"type": "any", "required": True},
        },
        output_schema={"df": {"type": "dataframe"}},
        handler=_handle_filter_rows,
    ),
    "group_by": Tool(
        name="group_by",
        description="Group and aggregate DataFrame",
        category=ToolCategory.DATA_TRANSFORM,
        granularity=ToolGranularity.FINE,
        input_schema={
            "df": {"type": "dataframe", "required": True},
            "group_cols": {"type": "list", "required": True},
            "agg_dict": {"type": "dict", "required": True},
        },
        output_schema={"df": {"type": "dataframe"}},
        handler=_handle_group_by,
    ),
    "select_columns": Tool(
        name="select_columns",
        description="Select specific columns from DataFrame",
        category=ToolCategory.DATA_TRANSFORM,
        granularity=ToolGranularity.FINE,
        input_schema={
            "df": {"type": "dataframe", "required": True},
            "columns": {"type": "list", "required": True},
        },
        output_schema={"df": {"type": "dataframe"}},
        handler=_handle_select_columns,
    ),
    "sort_data": Tool(
        name="sort_data",
        description="Sort DataFrame by columns",
        category=ToolCategory.DATA_TRANSFORM,
        granularity=ToolGranularity.FINE,
        input_schema={
            "df": {"type": "dataframe", "required": True},
            "by": {"type": "list", "required": True},
            "ascending": {"type": "bool", "required": False},
        },
        output_schema={"df": {"type": "dataframe"}},
        handler=_handle_sort_data,
    ),
    "fill_missing": Tool(
        name="fill_missing",
        description="Fill missing values in DataFrame",
        category=ToolCategory.DATA_TRANSFORM,
        granularity=ToolGranularity.FINE,
        input_schema={
            "df": {"type": "dataframe", "required": True},
            "strategy": {"type": "str", "required": False},
            "columns": {"type": "list", "required": False},
        },
        output_schema={"df": {"type": "dataframe"}},
        handler=_handle_fill_missing,
    ),
    
    # Data Transform - Coarse
    "pandas_transform": Tool(
        name="pandas_transform",
        description="Execute sequence of pandas operations",
        category=ToolCategory.DATA_TRANSFORM,
        granularity=ToolGranularity.COARSE,
        input_schema={
            "df": {"type": "dataframe", "required": True},
            "operations": {"type": "list", "required": True},
        },
        output_schema={"df": {"type": "dataframe"}},
        handler=_handle_pandas_transform,
        requires_llm=True,
    ),
    
    # Modeling - Coarse
    "custom_python": Tool(
        name="custom_python",
        description="Write and execute custom Python code for complex analysis or reporting",
        category=ToolCategory.MODELING,
        granularity=ToolGranularity.COARSE,
        input_schema={
            "df": {"type": "dataframe", "required": True},
            "query": {"type": "str", "required": True},
            "context": {"type": "str", "required": False},
        },
        output_schema={"result": {"type": "any"}, "code_generated": {"type": "str"}, "error": {"type": "str"}},
        handler=_handle_custom_python,
        requires_llm=True,
    ),
    
    # Visualization
    "chart_generator": Tool(
        name="chart_generator",
        description="Generate interactive chart",
        category=ToolCategory.VISUALIZATION,
        granularity=ToolGranularity.COARSE,
        input_schema={
            "df": {"type": "dataframe", "required": True},
            "chart_type": {"type": "str", "required": True},
            "x": {"type": "str", "required": True},
            "y": {"type": "str", "required": False},
        },
        output_schema={"figure": {"type": "plotly_figure"}},
        handler=_handle_chart_generator,
        fallback_tool="table_display",
    ),
    "table_display": Tool(
        name="table_display",
        description="Display data as table (fallback)",
        category=ToolCategory.VISUALIZATION,
        granularity=ToolGranularity.FINE,
        input_schema={
            "df": {"type": "dataframe", "required": True},
            "max_rows": {"type": "int", "required": False},
        },
        output_schema={"data": {"type": "list"}},
        handler=_handle_table_display,
    ),
}

# Category groupings
TOOL_CATEGORIES = {
    category: [name for name, tool in TOOL_REGISTRY.items() if tool.category == category]
    for category in ToolCategory
}


# =============================================================================
# Public API
# =============================================================================

def get_tool(name: str) -> Optional[Tool]:
    """Get a tool by name."""
    return TOOL_REGISTRY.get(name)


def list_tools(category: Optional[ToolCategory] = None, granularity: Optional[ToolGranularity] = None) -> List[str]:
    """List available tools with optional filtering."""
    tools = TOOL_REGISTRY.values()
    
    if category:
        tools = [t for t in tools if t.category == category]
    if granularity:
        tools = [t for t in tools if t.granularity == granularity]
    
    return [t.name for t in tools]


def invoke_tool(
    name: str,
    inputs: Dict[str, Any],
    max_retries: Optional[int] = None,
    use_fallback: bool = True,
) -> ToolResult:
    """
    Invoke a tool with retry logic and fallback support.
    
    Args:
        name: Tool name
        inputs: Tool inputs
        max_retries: Override tool's default max_retries
        use_fallback: Whether to try fallback tool on failure
        
    Returns:
        ToolResult with success/failure and output
    """
    tool = get_tool(name)
    if not tool:
        return ToolResult(success=False, error=f"Unknown tool: {name}")
    
    # Validate inputs
    is_valid, error_msg = tool.validate_inputs(inputs)
    if not is_valid:
        return ToolResult(success=False, error=error_msg)
    
    retries = max_retries if max_retries is not None else tool.max_retries
    last_error = None
    
    for attempt in range(retries):
        try:
            start_time = time.time()
            output = tool.handler(**inputs)
            execution_time = int((time.time() - start_time) * 1000)
            
            logger.debug(f"Tool '{name}' succeeded on attempt {attempt + 1}")
            return ToolResult(
                success=True,
                output=output,
                execution_time_ms=execution_time,
                metadata={"tool": name, "attempt": attempt + 1},
            )
            
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Tool '{name}' failed on attempt {attempt + 1}: {e}")
            
            # Exponential backoff
            if attempt < retries - 1:
                wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s...
                time.sleep(wait_time)
    
    # All retries exhausted - try fallback
    if use_fallback and tool.fallback_tool:
        logger.info(f"Trying fallback tool: {tool.fallback_tool}")
        return invoke_tool(tool.fallback_tool, inputs, use_fallback=False)
    
    return ToolResult(
        success=False,
        error=f"Tool '{name}' failed after {retries} attempts: {last_error}",
        metadata={"tool": name, "attempts": retries},
    )


def get_tool_schema(name: str) -> Optional[Dict[str, Any]]:
    """Get the input/output schema for a tool."""
    tool = get_tool(name)
    if not tool:
        return None
    
    return {
        "name": tool.name,
        "description": tool.description,
        "category": tool.category.value,
        "granularity": tool.granularity.value,
        "input_schema": tool.input_schema,
        "output_schema": tool.output_schema,
        "requires_llm": tool.requires_llm,
        "fallback": tool.fallback_tool,
    }
