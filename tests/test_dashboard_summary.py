import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_dummy_streamlit(summary_text):
    calls = []
    st = types.ModuleType("streamlit")
    st.session_state = {
        "result": {
            "analysis_type": "descriptive",
            "stats": None,
            "anomalies": None,
            "summary": summary_text,
        },
        "needs_role_review": False,
        "show_column_review": False,
    }

    st.sidebar = types.SimpleNamespace(
        file_uploader=lambda *a, **k: None,
        selectbox=lambda *a, **k: None,
        write=lambda *a, **k: calls.append(("sidebar.write", a, k)),
        subheader=lambda *a, **k: calls.append(("sidebar.subheader", a, k)),
        button=lambda *a, **k: False,
        markdown=lambda *a, **k: calls.append(("sidebar.markdown", a, k)),
        )

    def record(name):
        def fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return None
        return fn

    st.title = record("title")
    st.write = record("write")
    st.dataframe = record("dataframe")
    st.warning = record("warning")
    st.error = record("error")
    st.info = record("info")
    st.button = lambda *a, **k: False
    st.subheader = record("subheader")
    st.slider = lambda *a, **k: 3
    st.header = record("header")
    st.success = record("success")
    st.line_chart = record("line_chart")
    st.json = record("json")
    st.scatter_chart = record("scatter_chart")
    st.pyplot = record("pyplot")

    @contextmanager
    def spinner(*a, **kw):
        yield

    st.spinner = spinner
    return st, calls


def test_summary_display(monkeypatch):
    st, calls = _make_dummy_streamlit("Summary text")
    monkeypatch.setitem(sys.modules, "streamlit", st)

    monkeypatch.setitem(
        sys.modules,
        "feedback.ratings",
        types.SimpleNamespace(
            store_rating=lambda *a, **k: None,
            get_average_rating=lambda *a, **k: 0.0,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "feedback.role_corrections",
        types.SimpleNamespace(store_role_corrections=lambda *a, **k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "feedback.role_corrections",
        types.SimpleNamespace(store_role_corrections=lambda *a, **k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "ui.column_review",
        types.SimpleNamespace(column_review=lambda *a, **k: None),
    )

    importlib.import_module("ui.dashboard")

    assert any(c[0] == "subheader" and c[1][0] == "Business Summary" for c in calls)
    assert any(c[0] == "write" and c[1][0] == "Summary text" for c in calls)


def test_modeling_failure_message(monkeypatch):
    st, calls = _make_dummy_streamlit("Summary text")
    st.session_state["result"]["modeling_failed"] = True
    st.session_state["result"]["failure_reason"] = "Bad metrics"
    monkeypatch.setitem(sys.modules, "streamlit", st)

    monkeypatch.setitem(
        sys.modules,
        "feedback.ratings",
        types.SimpleNamespace(
            store_rating=lambda *a, **k: None,
            get_average_rating=lambda *a, **k: 0.0,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ui.column_review",
        types.SimpleNamespace(column_review=lambda *a, **k: None),
    )

    import importlib as _imp
    if "ui.dashboard" not in sys.modules:
        importlib.import_module("ui.dashboard")
    _imp.reload(sys.modules["ui.dashboard"])

    assert any(c[0] == "error" for c in calls)
    assert any(c[0] == "info" and "Bad metrics" in c[1][0] for c in calls)

