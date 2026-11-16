"""OpenVINO export (FP32, FP16, INT8) and latency / throughput on the CPU, iGPU and NPU of an Intel laptop."""

from __future__ import annotations

import csv
import json
import logging
import random
import shutil
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from road_damage.data import dataset_yaml

log = logging.getLogger(__name__)

PRECISIONS = {"fp32": 32, "fp16": 16, "int8": 8}


def read_list(path: Path) -> list[Path]:
    return [Path(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def calibration_yaml(cfg: dict[str, Any], per_country: int = 50) -> Path:
    """Dataset file whose val split is `per_country` random train images of every country, for INT8 calibration.

    Ultralytics' `fraction` takes the first images of the list, which would all come from the first country.
    """
    base = dataset_yaml(cfg)
    dataset = yaml.safe_load(base.read_text(encoding="utf-8"))
    picked = []
    for name in dataset["train"]:
        images = read_list(base.parent / name)
        rng = random.Random(f"{cfg['data']['seed']}-calibration-{name}")
        picked += rng.sample(images, min(per_country, len(images)))
    (base.parent / "calibration.txt").write_text("".join(f"{p.as_posix()}\n" for p in picked), "utf-8")
    dataset.update({"train": ["calibration.txt"], "val": ["calibration.txt"], "test": []})
    path = base.parent / "calibration.yaml"
    path.write_text(yaml.safe_dump(dataset, sort_keys=False), "utf-8")
    return path


def export(cfg: dict[str, Any], weights: str | Path, precision: str, calibration_images: int = 50) -> Path:
    """OpenVINO model of `weights` next to it, in <weights stem>_<precision>_openvino_model/."""
    from ultralytics import YOLO

    weights = Path(weights)
    args = {"format": "openvino", "imgsz": cfg["train"]["imgsz"], "quantize": PRECISIONS[precision]}
    if precision == "int8":
        args["data"] = str(calibration_yaml(cfg, calibration_images))
    out = Path(YOLO(str(weights)).export(**args))
    # Ultralytics gives FP32 and FP16 the same folder name (INT8 already has the right one)
    target = weights.parent / f"{weights.stem}_{precision}_openvino_model"
    if out.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        out.rename(target)
    return target


def letterbox(path: Path, size: int) -> np.ndarray:
    """1x3xHxW float32 input in [0, 1]: long side resized to `size`, padded with grey like Ultralytics."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        r = size / max(im.size)
        im = im.resize((round(im.width * r), round(im.height * r)), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return (np.asarray(canvas, dtype=np.float32) / 255).transpose(2, 0, 1)[None].copy()


def benchmark_openvino(model_dir: Path, device: str, inputs: Sequence[np.ndarray], runs: int = 200) -> dict[str, float]:
    """Latency of one image at a time (LATENCY hint) and images per second with parallel requests (THROUGHPUT)."""
    import openvino as ov

    core = ov.Core()
    model = core.read_model(next(Path(model_dir).glob("*.xml")))
    t0 = time.perf_counter()
    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
    compile_s = time.perf_counter() - t0
    request = compiled.create_infer_request()
    for x in inputs[:10]:
        request.infer({0: x})
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        request.infer({0: inputs[i % len(inputs)]})
        times.append((time.perf_counter() - t0) * 1000)

    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "THROUGHPUT"})
    queue = ov.AsyncInferQueue(compiled)
    for x in inputs[:10]:
        queue.start_async({0: x})
    queue.wait_all()
    t0 = time.perf_counter()
    for i in range(runs):
        queue.start_async({0: inputs[i % len(inputs)]})
    queue.wait_all()
    fps = runs / (time.perf_counter() - t0)
    return {**_latency(times), "throughput_fps": fps, "requests": len(queue), "compile_s": compile_s}


def benchmark_torch(weights: Path, inputs: Sequence[np.ndarray], runs: int = 50) -> dict[str, float]:
    """Same latency measure with the PyTorch model on the CPU, as a reference."""
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights)).model.float().eval()
    times = []
    with torch.inference_mode():
        for x in inputs[:3]:
            model(torch.from_numpy(x))
        for i in range(runs):
            t0 = time.perf_counter()
            model(torch.from_numpy(inputs[i % len(inputs)]))
            times.append((time.perf_counter() - t0) * 1000)
    return _latency(times)


def _latency(times: Sequence[float]) -> dict[str, float]:
    return {"median_ms": statistics.median(times), "p90_ms": float(np.quantile(times, 0.9)), "runs": len(times)}


def benchmark(
    cfg: dict[str, Any],
    weights: Path,
    models: Sequence[Path],
    devices: Sequence[str],
    images: int = 50,
    runs: int = 200,
) -> list[dict[str, Any]]:
    """PyTorch on the CPU, then every OpenVINO model on every device, on test images. Written next to `weights`."""
    imgsz = cfg["train"]["imgsz"]
    base = dataset_yaml(cfg)
    lists = yaml.safe_load(base.read_text(encoding="utf-8"))["test"]
    tests = [p for name in lists for p in read_list(base.parent / name)]
    paths = random.Random(cfg["data"]["seed"]).sample(tests, min(images, len(tests)))
    inputs = [letterbox(p, imgsz) for p in paths]
    rows = [{"model": weights.name, "device": "CPU (PyTorch)", **benchmark_torch(weights, inputs, runs=min(runs, 50))}]
    for model_dir in models:
        for device in devices:
            log.info("%s on %s", model_dir.name, device)
            result = benchmark_openvino(model_dir, device, inputs, runs)
            rows.append({"model": model_dir.name, "device": device, **result})
    run = weights.resolve().parent.parent
    (run / "benchmark.json").write_text(json.dumps(rows, indent=2), "utf-8")
    (run / "benchmark.md").write_text(benchmark_table(rows) + "\n", "utf-8")
    return rows


def benchmark_table(rows: Sequence[dict[str, Any]]) -> str:
    header = ["Model", "Device", "Latency median (ms)", "p90 (ms)", "Throughput (img/s)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        fps = f"{r['throughput_fps']:.1f}" if "throughput_fps" in r else "-"
        lines.append(f"| {r['model']} | {r['device']} | {r['median_ms']:.1f} | {r['p90_ms']:.1f} | {fps} |")
    return "\n".join(lines)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare_with_ultralytics(
    model_dir: Path, detections_csv: Path, images: Sequence[Path], conf: float = 0.25, iou: float = 0.7
) -> dict[str, int]:
    """Boxes of the C++ app (detections.csv of cpp/rdd_detect) against Ultralytics on the same OpenVINO model.

    A box matches when Ultralytics has one of the same class with an IoU above 0.99.
    """
    from ultralytics import YOLO

    imgsz = yaml.safe_load((model_dir / "metadata.yaml").read_text(encoding="utf-8"))["imgsz"]
    model = YOLO(str(model_dir), task="detect")
    cpp: dict[str, list[tuple[str, list[float]]]] = {}
    with detections_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cpp.setdefault(row["image"], []).append((row["class"], [float(row[k]) for k in ("x1", "y1", "x2", "y2")]))
    counts = {"images": 0, "cpp": 0, "ultralytics": 0, "matched": 0}
    for path in images:
        r = model.predict(str(path), imgsz=imgsz, conf=conf, iou=iou, device="intel:cpu", verbose=False)[0]
        ref = [(model.names[int(c)], b.tolist()) for c, b in zip(r.boxes.cls, r.boxes.xyxy, strict=True)]
        mine = cpp.get(path.name, [])
        counts["images"] += 1
        counts["cpp"] += len(mine)
        counts["ultralytics"] += len(ref)
        counts["matched"] += sum(any(c == rc and box_iou(b, rb) > 0.99 for rc, rb in ref) for c, b in mine)
    return counts
