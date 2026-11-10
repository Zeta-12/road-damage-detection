import json
from pathlib import Path

import pytest

from road_damage import data
from road_damage.train import markdown_table


def test_markdown_table_marks_missing_classes():
    results = {
        "all": {"map50": 0.5, "map50_95": 0.25, "precision": 0.6, "recall": 0.4, "ap50": {"D00": 0.5, "D40": 0.7}},
    }
    table = markdown_table(results, data.CLASSES)
    assert "| all | 0.500 | 0.250 | 0.600 | 0.400 | 0.500 | - | - | 0.700 |" in table


def test_markdown_table_adds_image_counts_of_subsets():
    r = {"map50": 0.5, "map50_95": 0.25, "precision": 0.6, "recall": 0.4, "ap50": {}}
    table = markdown_table({"all": {**r, "images": 12}, "Japan": r}, data.CLASSES)
    assert "| all | 12 | 0.500 |" in table
    assert "| Japan | - | 0.500 |" in table


def test_train_and_evaluate_end_to_end(prepared_cfg):
    # 1 epoch from scratch on the toy dataset, only checks that Ultralytics reads the labels, the split lists and the
    # dataset files
    pytest.importorskip("ultralytics")
    from road_damage.train import evaluate, train

    rdd_cfg = prepared_cfg
    rdd_cfg["train"].update({"model": "yolo11n.yaml", "epochs": 1, "imgsz": 64, "batch": 8, "workers": 0})
    rdd_cfg["train"].update({"device": "cpu", "extra": {"plots": False}})
    best = train(rdd_cfg, name="smoke")
    assert best.exists()
    assert not (best.parent.parent / "results.png").exists()  # extra reaches model.train()
    results = evaluate(rdd_cfg, best)
    assert set(results) == {"all", "Japan", "Norway"}
    saved = json.loads((best.parent.parent / "eval_test_64.json").read_text())
    assert saved["imgsz"] == 64
    assert 0.0 <= saved["all"]["map50"] <= 1.0

    # test images without a copy in train: all of Norway (its 6 grey levels are different)
    sub = evaluate(rdd_cfg, best, similarity=(0.0, 0.5))
    norway_test = (Path(rdd_cfg["data"]["root"]) / "yolo" / "Norway_test.txt").read_text().splitlines()
    assert sub["Norway"]["images"] == len(norway_test)
    assert (best.parent.parent / "eval_test_64_sim0-0.5.json").exists()

    # one country at another size: no "all" row, separate file
    rdd_cfg["data"]["countries"] = ["Norway"]
    rdd_cfg["train"]["imgsz"] = 96
    assert set(evaluate(rdd_cfg, best)) == {"Norway"}
    assert (best.parent.parent / "eval_test_96.json").exists()
