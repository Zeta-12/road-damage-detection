"""RDD2022: download, VOC to YOLO labels, train / val / test split per country."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

log = logging.getLogger(__name__)

# https://doi.org/10.6084/m9.figshare.21431547 (CC BY 4.0)
RDD_URL = "https://ndownloader.figshare.com/files/38030910"
RDD_MD5 = "b62bd51d2ffcfaa76c60f234f0cc2bb3"
RDD_ARCHIVE = "RDD2022_released_through_CRDDC2022.zip"

COUNTRIES = ("Japan", "India", "Norway", "United_States", "Czech", "China_MotorBike", "China_Drone")
CLASSES = ("D00", "D10", "D20", "D40")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Box:
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(root: Path, url: str = RDD_URL, md5: str = RDD_MD5) -> Path:
    """Download the archive next to `root` (skipped if it is already there), check the MD5, extract it."""
    archive = root.parent / RDD_ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        log.info("Downloading %s (13.3 GB) to %s", url, archive)
        tmp = archive.with_suffix(".part")
        with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, length=1 << 22)
        tmp.rename(archive)
    log.info("Checking MD5 of %s", archive)
    got = md5sum(archive)
    if got != md5:
        raise ValueError(f"MD5 mismatch for {archive}: {got}, expected {md5}. Delete it and download again.")
    extract(archive, root)
    return root


def extract(archive: Path, root: Path) -> None:
    """Extract the archive so that `root` contains one folder per country (nested zips are extracted too)."""
    work = root.parent / (root.name + "_extract")
    log.info("Extracting %s", archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(work)
    for inner in sorted(work.rglob("*.zip")):
        log.info("Extracting %s", inner.relative_to(work))
        with zipfile.ZipFile(inner) as z:
            z.extractall(inner.parent)
        inner.unlink()
    found = find_country_root(work)
    root.mkdir(parents=True, exist_ok=True)
    for country_dir in sorted(p for p in found.iterdir() if p.is_dir()):
        target = root / country_dir.name
        if target.exists():
            raise FileExistsError(f"{target} already exists")
        shutil.move(str(country_dir), str(target))
    shutil.rmtree(work)


def find_country_root(path: Path) -> Path:
    """First folder under `path` that contains at least one country folder with train/images."""
    for d in [path, *sorted(p for p in path.rglob("*") if p.is_dir())]:
        if any((d / c / "train" / "images").is_dir() for c in COUNTRIES):
            return d
    raise FileNotFoundError(f"No RDD2022 country folder under {path}")


def parse_voc(xml_path: Path) -> list[Box]:
    """All the boxes of a Pascal VOC file, whatever their label."""
    boxes = []
    for obj in ET.parse(xml_path).getroot().iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        # some countries store the coordinates as floats ("123.0")
        coords = [float(bb.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax")]
        boxes.append(Box((obj.findtext("name") or "").strip(), *coords))
    return boxes


def to_yolo(boxes: Sequence[Box], width: int, height: int, classes: Sequence[str]) -> tuple[list[str], Counter]:
    """YOLO label lines (class cx cy w h, normalised). Boxes are clipped to the image.

    The second value counts what was dropped: labels outside `classes` and empty boxes.
    """
    lines, dropped = [], Counter()
    for b in boxes:
        if b.label not in classes:
            dropped[f"label:{b.label}"] += 1
            continue
        x0, x1 = sorted((min(max(b.xmin, 0.0), width), min(max(b.xmax, 0.0), width)))
        y0, y1 = sorted((min(max(b.ymin, 0.0), height), min(max(b.ymax, 0.0), height)))
        if not np.isfinite([x0, x1, y0, y1]).all() or x1 - x0 < 1 or y1 - y0 < 1:
            dropped["empty_box"] += 1
            continue
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        w, h = (x1 - x0) / width, (y1 - y0) / height
        lines.append(f"{classes.index(b.label)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines, dropped


def index_country(root: Path, country: str) -> list[tuple[Path, Path | None]]:
    """(image, xml or None) for every labelled image of a country, in file name order."""
    images = sorted((root / country / "train" / "images").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images in {root / country / 'train' / 'images'}")
    xml_dir = root / country / "train" / "annotations" / "xmls"
    items = []
    for p in images:
        xml = xml_dir / p.with_suffix(".xml").name
        items.append((p, xml if xml.exists() else None))
    return items


def block_split(names: Sequence[str], ratios: Sequence[float], block_size: int, seed: str) -> dict[str, list[str]]:
    """Split sorted names into train / val / test by blocks of `block_size` consecutive names.

    Blocks are for datasets numbered in capture order, where consecutive frames are near-duplicates and a split per
    image would leak. `sequence_check` tells if that is the case. block_size=1 is a plain random split.
    """
    if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-6 or min(ratios) < 0:
        raise ValueError(f"split ratios must be 3 non-negative numbers summing to 1, got {ratios}")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    names = sorted(names)
    blocks = [names[i : i + block_size] for i in range(0, len(names), block_size)]
    order = list(range(len(blocks)))
    random.Random(seed).shuffle(order)
    n_train = round(ratios[0] * len(blocks))
    n_val = round(ratios[1] * len(blocks))
    parts = {"train": order[:n_train], "val": order[n_train : n_train + n_val], "test": order[n_train + n_val :]}
    return {split: sorted(n for i in idx for n in blocks[i]) for split, idx in parts.items()}


def prepare(cfg: dict[str, Any]) -> dict[str, Any]:
    """Write YOLO labels next to the images, the split lists and the Ultralytics dataset files.

    Labels go to <country>/train/labels/ (Ultralytics finds them by replacing /images/ with /labels/). Outputs in
    <root>/yolo/ (always, whatever data.split_dir): <country>_<split>.txt (absolute image paths), rdd2022.yaml (all
    countries), rdd2022_<country>.yaml, splits.json (image names per split) and stats.json.
    """
    d = cfg["data"]
    root = Path(d["root"]).resolve()
    classes = list(d["classes"])
    out = root / "yolo"
    out.mkdir(parents=True, exist_ok=True)
    splits, stats = {}, {}
    for country in d["countries"]:
        items = index_country(root, country)
        label_dir = root / country / "train" / "labels"
        label_dir.mkdir(exist_ok=True)
        s = {"images": len(items), "no_xml": 0, "background": 0, "boxes": Counter(), "dropped": Counter()}
        sizes = Counter()
        for image, xml in items:
            with Image.open(image) as im:
                width, height = im.size
            sizes[f"{width}x{height}"] += 1
            boxes = parse_voc(xml) if xml else []
            s["no_xml"] += xml is None
            lines, dropped = to_yolo(boxes, width, height, classes)
            s["background"] += not lines
            s["boxes"].update(classes[int(line.split()[0])] for line in lines)
            s["dropped"].update(dropped)
            (label_dir / image.with_suffix(".txt").name).write_text("".join(f"{x}\n" for x in lines), "utf-8")
        s["sizes"] = dict(sizes.most_common())
        s["boxes"] = {c: s["boxes"][c] for c in classes}
        s["dropped"] = dict(sorted(s["dropped"].items()))
        stats[country] = s
        splits[country] = block_split([p.name for p, _ in items], d["split"], d["block_size"], f"{d['seed']}-{country}")
        log.info("%s: %d images, %s", country, len(items), {k: len(v) for k, v in splits[country].items()})
    meta = {"split": d["split"], "block_size": d["block_size"], "seed": d["seed"]}
    write_split(root, out, splits, classes, meta)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), "utf-8")
    return stats


def write_split(
    root: Path, out: Path, splits: dict[str, dict[str, list[str]]], classes: Sequence[str], meta: dict[str, Any]
) -> None:
    """<country>_<split>.txt lists, rdd2022.yaml (all countries), rdd2022_<country>.yaml and splits.json in `out`.

    An empty split is left out of the dataset files (a country kept for training only has no val / test).
    """
    out.mkdir(parents=True, exist_ok=True)
    for country, parts in splits.items():
        images = root / country / "train" / "images"
        for split, names in parts.items():
            (out / f"{country}_{split}.txt").write_text("".join(f"{(images / n).as_posix()}\n" for n in names), "utf-8")
    countries = list(splits)
    for name, subset in [("rdd2022", countries), *[(f"rdd2022_{c}", [c]) for c in countries]]:
        dataset = {"path": out.as_posix()}
        dataset.update({split: [f"{c}_{split}.txt" for c in subset if splits[c][split]] for split in SPLITS})
        dataset["names"] = dict(enumerate(classes))
        (out / f"{name}.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), "utf-8")
    meta = {"classes": list(classes), **meta}
    (out / "splits.json").write_text(json.dumps({**meta, "splits": splits}, indent=1), "utf-8")


def dataset_yaml(cfg: dict[str, Any], country: str | None = None) -> Path:
    """Dataset file of data.split_dir: yolo/ is the random split of `rdd prepare`, see also `rdd group-split`."""
    name = "rdd2022" if country in (None, "all") else f"rdd2022_{country}"
    path = Path(cfg["data"]["root"]).resolve() / cfg["data"]["split_dir"] / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `rdd prepare` (or `rdd group-split`) first.")
    return path


def stats_table(stats: dict[str, Any], classes: Sequence[str]) -> str:
    header = ["Country", "Images", "No damage", *classes, "Dropped", "Main size"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for country, s in stats.items():
        dropped = sum(s["dropped"].values())
        main = next(iter(s["sizes"]), "-")
        row = [country, s["images"], s["background"], *(s["boxes"][c] for c in classes), dropped, main]
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def sequence_check(images: Sequence[Path], pairs: int = 200, seed: int = 0, size: int = 32) -> dict[str, float]:
    """Mean absolute difference between thumbnails of consecutive images and of random pairs.

    A ratio well below 1 means the images are frames of the same drive, so they must not be split at random.
    """
    images = sorted(images)
    if len(images) < 3:
        raise ValueError("need at least 3 images")
    rng = np.random.default_rng(seed)

    def thumb(p: Path) -> np.ndarray:
        with Image.open(p) as im:
            return np.asarray(im.convert("L").resize((size, size)), dtype=np.float32)

    starts = rng.choice(len(images) - 1, size=min(pairs, len(images) - 1), replace=False)
    consecutive = [np.abs(thumb(images[i]) - thumb(images[i + 1])).mean() for i in starts]
    shuffled = [rng.choice(len(images), size=2, replace=False) for _ in starts]
    random_pairs = [np.abs(thumb(images[i]) - thumb(images[j])).mean() for i, j in shuffled]
    c, r = float(np.mean(consecutive)), float(np.mean(random_pairs))
    return {"consecutive": c, "random": r, "ratio": c / r if r else float("nan")}
