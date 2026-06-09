import json

from dashboard import load_metrics


def test_load_metrics_handles_missing_file(tmp_path):
    assert load_metrics(tmp_path / "missing.json").empty


def test_load_metrics_reads_json(tmp_path):
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps([{"learn_cnt": 1, "total_loss": 0.5}]), encoding="utf-8")

    df = load_metrics(metrics_file)

    assert len(df) == 1
    assert df.iloc[0]["learn_cnt"] == 1
