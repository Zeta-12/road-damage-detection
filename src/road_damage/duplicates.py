"""Near-duplicates on CNN embeddings: nearest train image of every val / test image, and a split that keeps them
together."""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from road_damage.data import SPLITS, dataset_yaml, index_country, write_split

log = logging.getLogger(__name__)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class _Images(Dataset):
    def __init__(self, paths: Sequence[Path], size: int):
        self.paths, self.size = list(paths), size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as im:
            # big JPEGs (Norway) are decoded at a lower scale, much faster
            im.draft("RGB", (2 * self.size, 2 * self.size))
            im = im.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = (np.asarray(im, dtype=np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy())


def embed(
    paths: Sequence[Path],
    arch: str = "resnet50",
    weights: str | None = "DEFAULT",
    size: int = 224,
    batch: int = 64,
    workers: int = 4,
    device: str | None = None,
) -> np.ndarray:
    """L2-normalised pooled features of a torchvision ResNet (ImageNet weights), one row per image."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif str(device).isdigit():
        device = f"cuda:{device}"
    model = torchvision_model(arch, weights)
    model.eval().to(device)
    loader = DataLoader(_Images(paths, size), batch_size=batch, num_workers=workers, pin_memory=device != "cpu")
    out = []
    with torch.inference_mode():
        for x in loader:
            f = model(x.to(device, non_blocking=True))
            out.append(torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def torchvision_model(arch: str, weights: str | None) -> torch.nn.Module:
    import torchvision

    model = torchvision.models.get_model(arch, weights=weights)
    model.fc = torch.nn.Identity()
    return model


def nearest(query: np.ndarray, ref: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Index and cosine similarity of the closest `ref` row for every `query` row (rows already normalised)."""
    idx, sim = [], []
    for i in range(0, len(query), chunk):
        s = query[i : i + chunk] @ ref.T
        j = s.argmax(axis=1)
        idx.append(j)
        sim.append(s[np.arange(len(j)), j])
    return np.concatenate(idx), np.concatenate(sim)


def read_list(path: Path) -> list[Path]:
    return [Path(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def embeddings(
    cfg: dict[str, Any], arch: str = "resnet50", **embed_args: Any
) -> dict[str, tuple[list[str], np.ndarray]]:
    """(image names, embeddings) of every labelled image, per country.

    Cached in float16 in <root>/embeddings_<arch>.npz, which does not depend on the split.
    """
    root = Path(cfg["data"]["root"]).resolve()
    cache = root / f"embeddings_{arch}.npz"
    stored = dict(np.load(cache)) if cache.exists() else {}
    out, changed = {}, False
    for country in cfg["data"]["countries"]:
        paths = [p for p, _ in index_country(root, country)]
        names = [p.name for p in paths]
        pos = {n: i for i, n in enumerate(stored[f"{country}_names"].tolist())} if country in stored else {}
        if all(n in pos for n in names):
            e = stored[country][[pos[n] for n in names]].astype(np.float32)
        else:
            log.info("%s: embedding %d images", country, len(paths))
            e = embed(paths, arch=arch, **embed_args)
            stored[country], stored[f"{country}_names"] = e.astype(np.float16), np.array(names)
            changed = True
        out[country] = (names, e)
    if changed:
        np.savez(cache, **stored)
    return out


def find_duplicates(cfg: dict[str, Any], arch: str = "resnet50", **embed_args: Any) -> dict[str, Any]:
    """Nearest train image of every val and test image, per country, in the split of data.split_dir.

    Writes duplicates.json there: [image, nearest train image, cosine similarity] per val / test image.
    """
    out = Path(cfg["data"]["root"]).resolve() / cfg["data"]["split_dir"]
    emb = embeddings(cfg, arch, **embed_args)
    result = {}
    for country in cfg["data"]["countries"]:
        names, e = emb[country]
        pos = {n: i for i, n in enumerate(names)}
        lists = {split: [p.name for p in read_list(out / f"{country}_{split}.txt")] for split in SPLITS}
        train = e[[pos[n] for n in lists["train"]]]
        result[country] = {}
        for split in ("val", "test"):
            if not lists[split]:
                result[country][split] = []
                continue
            idx, sim = nearest(e[[pos[n] for n in lists[split]]], train)
            result[country][split] = [
                [n, lists["train"][j], float(s)] for n, j, s in zip(lists[split], idx, sim, strict=True)
            ]
    meta = {"arch": arch, "columns": ["image", "nearest_train", "similarity"]}
    (out / "duplicates.json").write_text(json.dumps({**meta, "countries": result}), "utf-8")
    return result


def components(e: np.ndarray, threshold: float, chunk: int = 2048) -> np.ndarray:
    """Group id of every row: rows linked by a cosine similarity >= threshold, directly or through other rows."""
    parent = np.arange(len(e))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, len(e), chunk):
        rows, cols = np.nonzero(e[start : start + chunk] @ e.T >= threshold)
        for i, j in zip(rows + start, cols, strict=True):
            a, b = find(i), find(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(len(e))])


def assign_groups(groups: np.ndarray, ratios: Sequence[float], seed: str) -> dict[str, np.ndarray]:
    """Row indices per split. Whole groups, biggest first, each one to the split furthest below its target size."""
    ids, sizes = np.unique(groups, return_counts=True)
    order = list(range(len(ids)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda k: -sizes[k])  # stable sort, groups of the same size stay shuffled
    rows = np.argsort(groups, kind="stable")
    starts = np.concatenate([[0], np.cumsum(sizes)])
    target, filled = np.array(ratios) * len(groups), np.zeros(len(SPLITS))
    parts = {split: [] for split in SPLITS}
    for k in order:
        s = int(np.argmax(target - filled))
        parts[SPLITS[s]].append(rows[starts[k] : starts[k + 1]])
        filled[s] += sizes[k]
    return {split: np.sort(np.concatenate(p)) if p else np.array([], dtype=int) for split, p in parts.items()}


def group_split(cfg: dict[str, Any], arch: str = "resnet50", **embed_args: Any) -> dict[str, Any]:
    """Train / val / test split that keeps near-duplicates together, written to <root>/<data.split_dir>.

    Groups are the connected components of the images whose embeddings have a cosine similarity of at least
    data.group_threshold. The countries of data.train_only only go to train.
    """
    d = cfg["data"]
    if d["split_dir"] == "yolo":
        raise ValueError("data.split_dir is yolo/, the random split of `rdd prepare`: use another folder")
    root = Path(d["root"]).resolve()
    emb = embeddings(cfg, arch, **embed_args)
    splits, report = {}, {}
    for country in d["countries"]:
        names, e = emb[country]
        if country in d["train_only"]:
            splits[country] = {"train": names, "val": [], "test": []}
            report[country] = {"images": len(names), "train_only": True}
            continue
        groups = components(e, d["group_threshold"])
        parts = assign_groups(groups, d["split"], f"{d['seed']}-{country}")
        splits[country] = {split: [names[i] for i in idx] for split, idx in parts.items()}
        sizes = np.unique(groups, return_counts=True)[1]
        report[country] = {"images": len(names), "groups": len(sizes), "largest": int(sizes.max())}
        report[country].update({split: len(idx) for split, idx in parts.items()})
        log.info("%s: %s", country, report[country])
    meta = {"split": d["split"], "group_threshold": d["group_threshold"], "train_only": d["train_only"], "arch": arch}
    write_split(root, root / d["split_dir"], splits, d["classes"], {**meta, "groups": report, "seed": d["seed"]})
    return report


def subset_yaml(
    cfg: dict[str, Any], country: str, split: str, similarity: tuple[float, float], out_dir: Path
) -> tuple[Path, int] | None:
    """Dataset file whose `split` only has the images with a nearest train similarity in [lo, hi).

    Needs duplicates.json (`rdd duplicates`). Returns the file and the number of images, None if no image is left.
    """
    lo, hi = similarity
    base = dataset_yaml(cfg, country)
    dataset = yaml.safe_load(base.read_text(encoding="utf-8"))
    rows = json.loads((base.parent / "duplicates.json").read_text(encoding="utf-8"))["countries"]
    out_dir.mkdir(parents=True, exist_ok=True)
    lists, n = [], 0
    for name in dataset[split]:
        keep = {image for image, _, s in rows[name.removesuffix(f"_{split}.txt")][split] if lo <= s < hi}
        paths = [p for p in read_list(base.parent / name) if p.name in keep]
        if paths:
            (out_dir / name).write_text("".join(f"{p.as_posix()}\n" for p in paths), "utf-8")
            lists.append((out_dir / name).as_posix())
            n += len(paths)
    if not n:
        return None
    dataset[split] = lists
    path = out_dir / base.name
    path.write_text(yaml.safe_dump(dataset, sort_keys=False), "utf-8")
    return path, n


def groups_table(report: dict[str, Any]) -> str:
    header = ["Country", "Images", "Groups", "Largest group", *SPLITS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for country, r in report.items():
        if r.get("train_only"):
            row = [country, r["images"], "-", "-", r["images"], 0, 0]
        else:
            row = [country, r["images"], r["groups"], r["largest"], *(r[s] for s in SPLITS)]
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def duplicates_table(result: dict[str, Any], split: str = "test", thresholds: Sequence[float] = (0.95, 0.98)) -> str:
    header = ["Country", "Images", "Median", "p90", "p99", "Max", *(f">= {t}" for t in thresholds)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for country, r in result.items():
        s = np.array([row[2] for row in r[split]])
        if not len(s):
            continue
        q = [f"{v:.3f}" for v in (np.median(s), np.quantile(s, 0.9), np.quantile(s, 0.99), s.max())]
        counts = [str(int((s >= t).sum())) for t in thresholds]
        lines.append("| " + " | ".join([country, str(len(s)), *q, *counts]) + " |")
    return "\n".join(lines)
