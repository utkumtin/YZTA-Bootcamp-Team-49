from io import BytesIO

import pandas as pd

from pareto.profiling import load_raw_file, profile_dataframe


def test_load_raw_file_accepts_streamlit_uploaded_file_like_csv():
    uploaded = BytesIO(b"a,b\n1,2\n3,4\n")
    uploaded.name = "deneme.csv"

    df = load_raw_file(uploaded)

    assert df.to_dict(orient="list") == {"a": [1, 3], "b": [2, 4]}


def test_load_raw_file_accepts_path(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text("a\tb\n1\t2\n", encoding="utf-8")

    df = load_raw_file(path)

    assert df.to_dict(orient="records") == [{"a": 1, "b": 2}]


def test_profile_dataframe_keeps_summary_only():
    df = pd.DataFrame({"state_id": [1, 2, 2], "name": ["a", "b", "b"]})

    profile = profile_dataframe(df)

    assert profile["n_rows"] == 3
    assert "state_id" in profile["potential_join_keys"]
    assert profile["duplicate_row_count"] == 1
