import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot.intent_parser import parse_intent
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_scatter_plot_intent():
    intent, params = parse_intent("Show me a scatter plot of sales vs profit")
    assert intent == "visualization"
    assert params["type"] == "scatter_plot"


def test_heat_map_intent():
    intent, params = parse_intent("Create a heat map for this dataset")
    assert intent == "visualization"
    assert params["type"] == "heat_map"


def test_pie_chart_intent():
    intent, params = parse_intent("I'd like a pie chart of categories")
    assert intent == "visualization"
    assert params["type"] == "pie_chart"


def test_forecast_intent():
    intent, params = parse_intent("Forecast next month's sales")
    assert intent == "modeling"
    assert params["task"] == "forecast"


def test_clustering_intent():
    intent, params = parse_intent("Could you cluster the customers?")
    assert intent == "modeling"
    assert params["task"] == "clustering"


def test_classification_intent():
    intent, params = parse_intent("Classify these transactions")
    assert intent == "modeling"
    assert params["task"] == "classification"


def test_llm_no_modeling(monkeypatch):
    monkeypatch.setattr(
        "chatbot.llm_intent_classifier.modeling_needed",
        lambda q: {"modeling_required": False, "reasoning": "desc"},
    )
    intent, params = parse_intent("Show the average sales")
    assert params.get("modeling_decision", {}).get("modeling_required") is False


def test_llm_with_modeling(monkeypatch):
    monkeypatch.setattr(
        "chatbot.llm_intent_classifier.modeling_needed",
        lambda q: {"modeling_required": True, "reasoning": "predict"},
    )
    monkeypatch.setattr(
        "chatbot.llm_intent_classifier.classify_modeling_type",
        lambda q: {"modeling_type": "regression", "reasoning": "because"},
    )
    intent, params = parse_intent("Predict sales")
    md = params.get("modeling_decision")
    assert md["modeling_required"] is True
    assert md["modeling_type"] == "regression"


def test_llm_failure(monkeypatch):
    def raise_exc(_):
        raise RuntimeError("fail")

    monkeypatch.setattr("chatbot.llm_intent_classifier.modeling_needed", raise_exc)
    result = parse_intent("Random text")
    assert result is None
