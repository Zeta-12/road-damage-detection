"""YAML configuration with defaults and command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "data": {
        "root": "datasets/RDD2022",
        "countries": ["Japan", "India", "Norway", "United_States", "Czech", "China_MotorBike", "China_Drone"],
        "classes": ["D00", "D10", "D20", "D40"],
        "split": [0.7, 0.1, 0.2],
        "block_size": 1,
        "seed": 42,
    },
    "train": {
        "model": "yolo26s.pt",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "workers": 4,
        "device": None,
    },
    "output_dir": "runs",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = DEFAULTS
    if path is not None:
        with Path(path).open(encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg
