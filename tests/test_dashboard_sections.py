import json
from pathlib import Path
import sys
import pandas as pd
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_dashboard_recommendations_and_report(tmp_path):
    dash_path = Path(__file__).resolve().parents[1] / "ui" / "dashboard.py"
    at = AppTest.from_file(str(dash_path))
    dummy_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    at.session_state["datasets"] = {"dummy": dummy_df}
    at.session_state["primary_dataset_id"] = "dummy"
    at.session_state["result"] = {
        "analysis_type": "regression",
        "predictions": [1, 2],
        "anomalies": [],
        "recommended_models": {"semantic_merge": ["lr", "rf"]},
        "model_info": {"merge_report": {"chosen_table": "dummy.csv"}},
    }
    at.run(timeout=20)

    exp = next(e for e in at.expander if e.label == "Recommended Models")
    assert exp.label == "Recommended Models"
    rec = json.loads(exp.json[0].value)
    assert rec["semantic_merge"] == ["lr", "rf"]

    report_exp = next(e for e in at.expander if e.label == "Merge Report")
    assert report_exp.label == "Merge Report"
    dl = report_exp.get("download_button")[0]
    assert dl.proto.label == "Download merge_report.json"


def test_run_history_display(tmp_path, monkeypatch):
    run_history = [
        {"run_id": "abc123", "score_ok": True, "needs_role_review": False}
    ]
    (tmp_path / "run_history.json").write_text(json.dumps(run_history))
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))

    dash_path = Path(__file__).resolve().parents[1] / "ui" / "dashboard.py"
    at = AppTest.from_file(str(dash_path))
    dummy_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    at.session_state["datasets"] = {"dummy": dummy_df}
    at.session_state["primary_dataset_id"] = "dummy"
    at.run(timeout=20)

    # Run History is now in an expander in the Analyze tab (not the sidebar)
    assert any(e.label == "Run History" for e in at.expander)
    all_text = "".join(m.value for m in at.markdown)
    assert "abc123" in all_text
