"""Simple intent parser with optional spaCy usage.

The parser relies primarily on simple keyword matching but, when spaCy is
available, a :class:`~spacy.matcher.PhraseMatcher` is used for slightly more
robust intent detection. This keeps the implementation lightweight while still
allowing more flexible matching when the dependency is installed.
"""

import logging
import warnings
from typing import Dict, Tuple

warnings.warn(
    "chatbot.intent_parser is deprecated and will be removed. "
    "Use orchestration.cascade_planner.classify_intent instead.",
    DeprecationWarning,
    stacklevel=2
)

from utils.logging import log_decision

from . import llm_intent_classifier

# Lazy loaded globals
nlp = None
_matcher = None

def _load_spacy():
    global nlp, _matcher, _phrases
    if nlp is not None:
        return nlp, _matcher
    
    try:
        import spacy
        from spacy.matcher import PhraseMatcher
        nlp = spacy.blank("en")
        _matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        
        _phrases = {
            "line_chart": ("line chart",),
            "bar_chart": ("bar chart", "compare"),
            "scatter_plot": ("scatter plot",),
            "heat_map": ("heat map", "heatmap"),
            "pie_chart": ("pie chart",),
            "forecast": ("forecast", "forecasting"),
            "clustering": ("cluster", "clustering"),
            "classification": ("classify", "classification"),
        }
        for key, patterns in _phrases.items():
            _matcher.add(key.upper(), [nlp.make_doc(p) for p in patterns])
    except Exception:
        nlp = False  # Sentinel for failed load
        _matcher = None
    
    return nlp, _matcher


def parse_intent(user_query: str):
    """Parse user queries into intents and parameters.

    Returns ``None`` if the LLM is unavailable so callers can handle the
    failure explicitly.
    """

    text = user_query.lower()
    lt = user_query.lower()

    intent = "unknown"
    params: Dict[str, str] = {}

    # Use spaCy matcher when available for more flexible detection
    # Use spaCy matcher when available for more flexible detection
    _nlp, _match = _load_spacy()
    if _match is not None:
        doc = _nlp(text)
        for match_id, start, end in _match(doc):
            label = _nlp.vocab.strings[match_id]
            if label == "LINE_CHART":
                intent, params = "visualization", {"type": "line_chart"}
                break
            if label == "BAR_CHART":
                intent, params = "visualization", {"type": "bar_chart"}
                break
            if label == "SCATTER_PLOT":
                intent, params = "visualization", {"type": "scatter_plot"}
                break
            if label == "HEAT_MAP":
                intent, params = "visualization", {"type": "heat_map"}
                break
            if label == "PIE_CHART":
                intent, params = "visualization", {"type": "pie_chart"}
                break
            if label == "FORECAST":
                intent, params = "modeling", {"task": "forecast"}
                break
            if label == "CLUSTERING":
                intent, params = "modeling", {"task": "clustering"}
                break
            if label == "CLASSIFICATION":
                intent, params = "modeling", {"task": "classification"}
                break

    if intent == "unknown":
        if "line" in text and "chart" in text:
            intent, params = "visualization", {"type": "line_chart"}
        elif "bar" in text or "compare" in text:
            intent, params = "visualization", {"type": "bar_chart"}
        elif "scatter" in text and "plot" in text:
            intent, params = "visualization", {"type": "scatter_plot"}
        elif "heat" in text and "map" in text:
            intent, params = "visualization", {"type": "heat_map"}
        elif "pie" in text and "chart" in text:
            intent, params = "visualization", {"type": "pie_chart"}
        elif "forecast" in text:
            intent, params = "modeling", {"task": "forecast"}
        elif "cluster" in text:
            intent, params = "modeling", {"task": "clustering"}
        elif "classify" in text or "classification" in text:
            intent, params = "modeling", {"task": "classification"}
        elif "simulate" in text or "scenario" in text:
            intent, params = "scenario", {"adjustments": {}}

    # Query LLM for modeling decision
    try:
        decision = llm_intent_classifier.modeling_needed(user_query)
        if isinstance(decision, dict) and decision.get("status") == "error":
            logging.getLogger(__name__).warning("%s", decision.get("reason"))
            if intent == "unknown":
                return None
            decision = None
        if decision and decision.get("modeling_required"):
            result = llm_intent_classifier.classify_modeling_type(user_query)
            if isinstance(result, dict) and result.get("status") == "error":
                logging.getLogger(__name__).warning("%s", result.get("reason"))
                if intent == "unknown":
                    return None
                result = None
            if result:
                decision.update(result)
        if decision:
            params["modeling_decision"] = decision
            log_decision(user_query, decision)
    except Exception as exc:  # pragma: no cover - LLM optional
        logging.getLogger(__name__).warning("LLM classification failed: %s", exc)
        if intent == "unknown":
            return None

    return intent, params

