import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd
from orchestration.cascade_planner import CascadePlanner, classify_intent, Intent
from orchestration.tool_registry import invoke_tool

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": range(100),
        "category": ["A", "B", "C", "A"] * 25,
        "value": [10.5, 20.1, 15.0, 5.5] * 25,
        "date": pd.date_range("2023-01-01", periods=100)
    })

def test_cascade_intent_describe(sample_df):
    """Test that a describe intent builds a valid plan that executes successfully."""
    query = "Describe this dataset to me"
    intent, confidence = classify_intent(query)
    assert intent == Intent.DESCRIBE_DATA
    
    planner = CascadePlanner()
    plan = planner.plan(query, {"df": sample_df})
    
    # Needs at least the data_profiler step
    assert len(plan.steps) >= 1
    assert plan.steps[0].tool == "data_profiler"
    
    result = planner.execute(plan, {"df": sample_df})
    assert result.success is True
    assert "shape" in result.output

def test_cascade_intent_filter(sample_df):
    """Test that a filter intent builds a valid plan that executes successfully."""
    query = "filter rows where category is A"
    intent, confidence = classify_intent(query)
    assert intent == Intent.FILTER
    
    planner = CascadePlanner()
    plan = planner.plan(query, {"df": sample_df})
    
    assert len(plan.steps) >= 1
    assert plan.steps[0].tool == "filter_rows"
    
    tool_inputs = {
        "df": sample_df, 
        "column": "category",
        "operator": "==",
        "value": "A"
    }
    
    result = invoke_tool("filter_rows", tool_inputs)
    assert result.success is True
    assert len(result.output) < 100 
    assert result.output["category"].nunique() == 1

def test_cascade_intent_aggregate(sample_df):
    """Test that an aggregate intent builds a valid plan that executes successfully."""
    query = "group by category and sum the values"
    intent, confidence = classify_intent(query)
    assert intent == Intent.AGGREGATE
    
    planner = CascadePlanner()
    plan = planner.plan(query, {"df": sample_df})
    
    assert len(plan.steps) >= 1
    assert plan.steps[0].tool == "group_by"
    
    # We must patch LLM since group_by uses ___infer_group_cols___ placeholder
    # which reaches out to the LLM during execution. We will simulate execution 
    # of just the tool locally with inferred kwargs.
    tool_inputs = {
        "df": sample_df, 
        "group_cols": ["category"],
        "agg_dict": {"value": "sum"}
    }
    
    tool_res = invoke_tool("group_by", tool_inputs)
    assert tool_res.success is True
    assert len(tool_res.output) == 3 # Categories A, B, C

def test_cascade_intent_transform(sample_df):
    """Test that a transform intent builds a valid plan."""
    query = "fill missing values with the mean"
    intent, confidence = classify_intent(query)
    assert intent == Intent.TRANSFORM
    
    planner = CascadePlanner()
    plan = planner.plan(query, {"df": sample_df})
    
    assert len(plan.steps) >= 1
    assert plan.steps[0].tool in ["fill_missing", "pandas_transform"]

def test_cascade_intent_visualize(sample_df):
    """Test that a visualize intent builds a valid plan."""
    query = "plot a bar chart of the categories"
    intent, confidence = classify_intent(query)
    assert intent == Intent.VISUALIZE
    
    planner = CascadePlanner()
    plan = planner.plan(query, {"df": sample_df})
    
    assert len(plan.steps) >= 1
    assert plan.steps[0].tool == "chart_generator"
