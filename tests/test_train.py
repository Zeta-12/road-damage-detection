import json

import pytest

from road_damage import data
from road_damage.train import markdown_table


def test_markdown_table_marks_missing_classes():
    results = {
        "all": {"map50": 0.5, "map50_95": 0.25, "precision": 0.6, "recall": 0.4, "ap50": {"D00": 0.5, "D40": 0.7}},
    }
    table = markdown_table(results, data.CLASSES)
    assert "| all | 0.500 | 0.250 | 0.600 | 0.400 | 0.500 | - | - | 0.700 |" in table


def test_train_and_evaluate_end_to_end(rdd_cfg, tmp_path):
    # 1 epoch from scratch on the toy dataset, only checks that Ultralytics reads the labels, the split lists and the
    # dataset files
    pytest.importorskip("ultralytics")
    from road_damage.train import evaluate, train

    data.prepare(rdd_cfg)
    rdd_cfg["train"].update({"model": "yolo11n.yaml", "epochs": 1, "imgsz": 64, "batch": 8, "workers": 0})
    rdd_cfg["train"]["device"] = "cpu"
    best = train(rdd_cfg, name="smoke")
    assert best.exists()
    results = evaluate(rdd_cfg, best)
    assert set(results) == {"all", "Japan", "Norway"}
    saved = json.loads((best.parent.parent / "eval_test.json").read_text())
    assert 0.0 <= saved["all"]["map50"] <= 1.0
