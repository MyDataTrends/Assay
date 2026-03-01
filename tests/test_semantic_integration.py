import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catalog.semantic_index import build_index
import Integration.semantic_merge as si
from preprocessing.metadata_parser import infer_column_meta


def _setup(tmp_path: Path, df: pd.DataFrame, fname: str):
    ddir = tmp_path / "datasets"
    ddir.mkdir()
    df.to_csv(ddir / fname, index=False)
    index_path = tmp_path / "semantic_index.db"
    build_index(str(ddir), db_path=index_path)
    si._DATASETS_DIR = ddir
    si._INDEX_DB = index_path
    return ddir, index_path


def test_direct_join(tmp_path):
    user_df = pd.DataFrame({"StoreID": [1, 2], "sales_amount": [10, 20]})
    other_df = pd.DataFrame({"StoreID": [1, 2], "Other": [5, 6]})
    _setup(tmp_path, other_df, "store.csv")

    meta = infer_column_meta(user_df)
    merged, report = si.rank_and_merge(user_df, meta, datasets_dir=si._DATASETS_DIR)
    assert "Other" in merged.columns
    assert report["chosen_table"] == "store.csv"


def test_hash_join(tmp_path):
    user_df = pd.DataFrame({"City": ["A", "B"], "State": ["X", "Y"], "sales": [1, 2]})
    pop_df = pd.DataFrame({"City": ["A", "B"], "State": ["X", "Y"], "Pop": [100, 200]})
    _setup(tmp_path, pop_df, "pop.csv")

    meta = infer_column_meta(user_df)
    merged, report = si.rank_and_merge(user_df, meta, datasets_dir=si._DATASETS_DIR)
    assert "Pop" in merged.columns
    assert report["chosen_table"] == "pop.csv"


def test_no_join(tmp_path):
    user_df = pd.DataFrame({"Foo": [1, 2]})
    other_df = pd.DataFrame({"Bar": [1, 2]})
    _setup(tmp_path, other_df, "other.csv")

    meta = infer_column_meta(user_df)
    merged, report = si.rank_and_merge(user_df, meta, datasets_dir=si._DATASETS_DIR)
    assert list(merged.columns) == ["Foo"]
    assert report["chosen_table"] is None

