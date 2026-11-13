import numpy as np
import pytest
import yaml
from PIL import Image

from road_damage import deploy


def test_calibration_yaml_takes_train_images_of_every_country(prepared_cfg, rdd_root):
    path = deploy.calibration_yaml(prepared_cfg, per_country=5)
    dataset = yaml.safe_load(path.read_text())
    assert dataset["val"] == ["calibration.txt"]
    picked = deploy.read_list(path.parent / "calibration.txt")
    assert sorted(p.name.split("_")[0] for p in picked) == ["Japan"] * 5 + ["Norway"] * 5
    train = {p for c in ("Japan", "Norway") for p in deploy.read_list(rdd_root / "yolo" / f"{c}_train.txt")}
    assert set(picked) <= train
    assert deploy.read_list(deploy.calibration_yaml(prepared_cfg, 5).parent / "calibration.txt") == picked


def test_letterbox_keeps_the_ratio_and_pads_with_grey(tmp_path):
    Image.new("RGB", (200, 100), (255, 0, 0)).save(tmp_path / "a.png")
    x = deploy.letterbox(tmp_path / "a.png", 64)
    assert x.shape == (1, 3, 64, 64)
    assert x.dtype == np.float32
    np.testing.assert_allclose(x[0, :, 0, 0], 114 / 255)  # padding above the image
    np.testing.assert_allclose(x[0, :, 32, 32], [1, 0, 0])


def test_benchmark_table():
    rows = [
        {"model": "best.pt", "device": "CPU (PyTorch)", "median_ms": 10.0, "p90_ms": 12.0, "runs": 5},
        {"model": "m", "device": "CPU", "median_ms": 3.0, "p90_ms": 4.0, "throughput_fps": 500.0},
    ]
    table = deploy.benchmark_table(rows)
    assert "| best.pt | CPU (PyTorch) | 10.0 | 12.0 | - |" in table
    assert "| m | CPU | 3.0 | 4.0 | 500.0 |" in table


def test_export_benchmark_and_evaluate_openvino(prepared_cfg):
    # 1 epoch on the toy dataset again, then the FP32 and INT8 exports, latency and mAP of the INT8 model
    pytest.importorskip("ultralytics")
    pytest.importorskip("openvino")
    pytest.importorskip("nncf")
    from road_damage.train import evaluate, train

    cfg = prepared_cfg
    cfg["train"].update({"model": "yolo11n.yaml", "epochs": 1, "imgsz": 64, "batch": 8, "workers": 0})
    cfg["train"].update({"device": "cpu", "extra": {"plots": False}})
    best = train(cfg, name="smoke")
    models = [deploy.export(cfg, best, precision, calibration_images=8) for precision in ("fp32", "int8")]
    assert [m.name for m in models] == ["best_fp32_openvino_model", "best_int8_openvino_model"]
    assert all(any(m.glob("*.xml")) for m in models)

    rows = deploy.benchmark(cfg, best, models, ["CPU"], images=4, runs=5)
    assert [r["device"] for r in rows] == ["CPU (PyTorch)", "CPU", "CPU"]
    assert all(r["median_ms"] > 0 for r in rows)
    assert (best.parent.parent / "benchmark.md").exists()

    # on the same square inputs, the FP32 export gives the mAP of the PyTorch model
    pytorch = evaluate(cfg, best, tag="pt_square", square=True)
    cfg["train"]["device"] = "intel:cpu"
    fp32 = evaluate(cfg, models[0], tag="fp32_cpu")
    assert abs(pytorch["all"]["map50"] - fp32["all"]["map50"]) < 0.01
    assert set(evaluate(cfg, models[1], tag="int8_cpu")) == {"all", "Japan", "Norway"}
    assert (best.parent.parent / "eval_test_64_int8_cpu.json").exists()
