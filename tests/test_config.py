from pathlib import Path

import pytest
import yaml

from road_damage.config import DEFAULTS, load_config

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))


def unknown_keys(cfg: dict, ref: dict, prefix: str = "") -> list[str]:
    bad = []
    for key, value in cfg.items():
        if key not in ref:
            bad.append(prefix + key)
        elif isinstance(value, dict) and ref[key]:  # an empty dict in DEFAULTS is free-form (train.extra)
            bad += unknown_keys(value, ref[key], f"{prefix}{key}.")
    return bad


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_files_only_use_known_keys(path):
    assert unknown_keys(yaml.safe_load(path.read_text(encoding="utf-8")), DEFAULTS) == []
    assert load_config(path)["data"]["classes"] == ["D00", "D10", "D20", "D40"]


def test_overrides_merge_into_nested_dicts():
    cfg = load_config(None, {"train": {"extra": {"lr0": 0.001}}})
    assert cfg["train"]["extra"] == {"lr0": 0.001}
    assert cfg["train"]["epochs"] == DEFAULTS["train"]["epochs"]
    assert DEFAULTS["train"]["extra"] == {}
