from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from road_damage.config import load_config

OBJECT = "<object><name>{}</name><bndbox><xmin>{}</xmin><ymin>{}</ymin><xmax>{}</xmax><ymax>{}</ymax></bndbox></object>"


def write_voc(path: Path, name: str, width: int, height: int, boxes) -> None:
    objects = "".join(OBJECT.format(*b) for b in boxes)
    path.write_text(
        f"<annotation><filename>{name}</filename><size><width>{width}</width><height>{height}</height>"
        f"<depth>3</depth></size>{objects}</annotation>",
        encoding="utf-8",
    )


def make_country(root: Path, country: str, n: int, width: int = 64, height: int = 48) -> None:
    images = root / country / "train" / "images"
    xmls = root / country / "train" / "annotations" / "xmls"
    images.mkdir(parents=True)
    xmls.mkdir(parents=True)
    for i in range(n):
        name = f"{country}_{i:06d}.jpg"
        # 10 consecutive images look the same, like frames of one drive
        value = 30 + 20 * ((i // 10) % 10)
        Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8)).save(images / name)
        boxes = [("D00", 5.0, 5.0, 20.0, 30.0)] if i % 3 else []
        if i == 1:
            # a label outside the 4 classes and a box past the right edge
            boxes += [("D43", 1, 1, 10, 10), ("D40", 50, 30, 80, 40)]
        if i == 2:
            continue  # no xml at all
        write_voc(xmls / f"{country}_{i:06d}.xml", name, width, height, boxes)


@pytest.fixture
def rdd_root(tmp_path: Path) -> Path:
    root = tmp_path / "RDD2022"
    make_country(root, "Japan", 120)
    make_country(root, "Norway", 60)
    return root


@pytest.fixture
def rdd_cfg(rdd_root: Path, tmp_path: Path) -> dict:
    return load_config(
        None,
        {
            "data": {"root": str(rdd_root), "countries": ["Japan", "Norway"], "block_size": 10},
            "output_dir": str(tmp_path / "runs"),
        },
    )
