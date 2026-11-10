"""Training and per-country evaluation with Ultralytics (YOLO, RT-DETR)."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from road_damage.data import dataset_yaml

log = logging.getLogger(__name__)


def load_model(weights: str | Path):
    from ultralytics import RTDETR, YOLO

    model = YOLO(str(weights))
    # RT-DETR needs its own validator (no NMS), and a best.pt doesn't say which family it is
    if type(model.model).__name__ == "RTDETRDetectionModel":
        return RTDETR(str(weights))
    return model


def train(cfg: dict[str, Any], country: str | None = None, name: str | None = None) -> Path:
    """Fine-tune `train.model` on all countries (or one). Returns the path of best.pt."""
    t = cfg["train"]
    data = dataset_yaml(cfg, country)
    name = name or f"{Path(t['model']).stem}_{country or 'all'}"
    model = load_model(t["model"])
    model.train(
        data=str(data),
        epochs=t["epochs"],
        imgsz=t["imgsz"],
        batch=t["batch"],
        workers=t["workers"],
        device=t["device"],
        seed=cfg["data"]["seed"],
        project=str(Path(cfg["output_dir"]).resolve()),
        name=name,
        **t["extra"],
    )
    save_dir = Path(model.trainer.save_dir)
    (save_dir / "rdd_config.json").write_text(json.dumps({"country": country, **cfg}, indent=2), "utf-8")
    return Path(model.trainer.best)


def evaluate(
    cfg: dict[str, Any], weights: str | Path, split: str = "test", similarity: tuple[float, float] | None = None
) -> dict[str, Any]:
    """mAP on `split` for each country, and for all of them together when there is more than one.

    "all" is every country of `rdd prepare`. `similarity` = (lo, hi) only keeps the images whose nearest train image
    has a cosine similarity in [lo, hi) (see `rdd duplicates`). Results go next to the weights' run, in
    eval_<split>_<imgsz>[_sim<lo>-<hi>].json / .md.
    """
    classes = cfg["data"]["classes"]
    imgsz = cfg["train"]["imgsz"]
    countries = list(cfg["data"]["countries"])
    tag = f"{split}_{imgsz}" + (f"_sim{similarity[0]:g}-{similarity[1]:g}" if similarity else "")
    run = Path(weights).resolve().parent.parent
    model = load_model(weights)
    results = {}
    for country in countries if len(countries) == 1 else ["all", *countries]:
        data, images = dataset_yaml(cfg, country), None
        if similarity:
            from road_damage.duplicates import subset_yaml

            sub = subset_yaml(cfg, country, split, similarity, run / f"data_{tag}")
            if sub is None:
                log.info("%s: no image in this similarity range", country)
                continue
            data, images = sub
        m = model.val(
            data=str(data),
            split=split,
            imgsz=imgsz,
            batch=cfg["train"]["batch"],
            workers=cfg["train"]["workers"],
            device=cfg["train"]["device"],
            project=str(run),
            name=f"val_{tag}_{country}",
            exist_ok=True,
            plots=country == "all",
            verbose=False,
        )
        results[country] = {
            "map50": float(m.box.map50),
            "map50_95": float(m.box.map),
            "precision": float(m.box.mp),
            "recall": float(m.box.mr),
            # classes with no instance in this split are missing
            "ap50": {classes[int(c)]: float(ap) for c, ap in zip(m.box.ap_class_index, m.box.ap50, strict=True)},
            "speed_ms": {k: float(v) for k, v in m.speed.items()},
        }
        if images is not None:
            results[country]["images"] = images
        log.info("%-16s mAP50 %.3f  mAP50-95 %.3f", country, results[country]["map50"], results[country]["map50_95"])
    meta = {"weights": str(weights), "imgsz": imgsz, "similarity": similarity}
    (run / f"eval_{tag}.json").write_text(json.dumps({**meta, **results}, indent=2), "utf-8")
    (run / f"eval_{tag}.md").write_text(markdown_table(results, classes) + "\n", "utf-8")
    return results


def markdown_table(results: dict[str, Any], classes: Sequence[str]) -> str:
    # image counts are only there for subsets
    counts = any("images" in r for r in results.values())
    metrics = ["mAP50", "mAP50-95", "P", "R", *(f"AP50 {c}" for c in classes)]
    header = ["Test set", *(["Images"] if counts else []), *metrics]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for country, r in results.items():
        ap = [f"{r['ap50'][c]:.3f}" if c in r["ap50"] else "-" for c in classes]
        row = [f"{r['map50']:.3f}", f"{r['map50_95']:.3f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}", *ap]
        if counts:
            row.insert(0, str(r.get("images", "-")))
        lines.append(f"| {country} | " + " | ".join(row) + " |")
    return "\n".join(lines)
