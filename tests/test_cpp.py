import os
import subprocess

import pytest

from road_damage import deploy

# path of the built cpp/rdd_detect, see the README
BINARY = os.environ.get("RDD_DETECT")


@pytest.mark.skipif(not BINARY, reason="RDD_DETECT is not set")
def test_cpp_app_gives_the_boxes_of_ultralytics(prepared_cfg, rdd_root, tmp_path):
    pytest.importorskip("openvino")
    from road_damage.train import train

    cfg = prepared_cfg
    cfg["train"].update({"model": "yolo11n.yaml", "epochs": 1, "imgsz": 64, "batch": 8, "workers": 0})
    cfg["train"].update({"device": "cpu", "extra": {"plots": False}})
    model = deploy.export(cfg, train(cfg, name="smoke"), "fp32")

    # the toy Norway images are 64x48, so the letterbox adds a border
    images = rdd_root / "Norway" / "train" / "images"
    out = tmp_path / "cpp"
    # very low threshold: the 1-epoch model is sure of nothing, this still gives boxes to compare
    cmd = [BINARY, "--model", str(model / "best.xml"), "--input", str(images), "--output", str(out), "--conf", "0.001"]
    subprocess.run(cmd, check=True)
    assert len(list(out.glob("*.jpg"))) == 60

    counts = deploy.compare_with_ultralytics(model, out / "detections.csv", sorted(images.glob("*.jpg")), conf=0.001)
    assert counts["cpp"] > 0
    assert counts["matched"] == counts["cpp"] == counts["ultralytics"]
