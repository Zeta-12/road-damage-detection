"""Near-duplicates between splits: nearest train image of every val / test image, on CNN embeddings."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from road_damage.data import SPLITS, dataset_yaml

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


def find_duplicates(cfg: dict[str, Any], arch: str = "resnet50", **embed_args: Any) -> dict[str, Any]:
    """Nearest train image of every val and test image, per country (after `rdd prepare`).

    Writes <root>/yolo/duplicates.json ([image, nearest train image, cosine similarity] per val / test image) and
    the embeddings in float16 (embeddings_<arch>.npz) to look at the threshold without running the model again.
    """
    out = Path(cfg["data"]["root"]).resolve() / "yolo"
    result, arrays = {}, {}
    for country in cfg["data"]["countries"]:
        paths = {split: read_list(out / f"{country}_{split}.txt") for split in SPLITS}
        everything = [p for split in SPLITS for p in paths[split]]
        log.info("%s: embedding %d images", country, len(everything))
        e = embed(everything, arch=arch, **embed_args)
        ends = np.cumsum([len(paths[split]) for split in SPLITS])
        parts = dict(zip(SPLITS, np.split(e, ends[:-1]), strict=True))
        train_names = [p.name for p in paths["train"]]
        result[country] = {}
        for split in ("val", "test"):
            idx, sim = nearest(parts[split], parts["train"])
            result[country][split] = [
                [p.name, train_names[j], float(s)] for p, j, s in zip(paths[split], idx, sim, strict=True)
            ]
        arrays[country] = e.astype(np.float16)
        arrays[f"{country}_names"] = np.array([p.name for p in everything])
    meta = {"arch": arch, "columns": ["image", "nearest_train", "similarity"]}
    (out / "duplicates.json").write_text(json.dumps({**meta, "countries": result}), "utf-8")
    np.savez(out / f"embeddings_{arch}.npz", **arrays)
    return result


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


def duplicates_table(result: dict[str, Any], split: str = "test", thresholds: Sequence[float] = (0.95, 0.98)) -> str:
    header = ["Country", "Images", "Median", "p90", "p99", "Max", *(f">= {t}" for t in thresholds)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for country, r in result.items():
        s = np.array([row[2] for row in r[split]])
        q = [f"{v:.3f}" for v in (np.median(s), np.quantile(s, 0.9), np.quantile(s, 0.99), s.max())]
        counts = [str(int((s >= t).sum())) for t in thresholds]
        lines.append("| " + " | ".join([country, str(len(s)), *q, *counts]) + " |")
    return "\n".join(lines)
