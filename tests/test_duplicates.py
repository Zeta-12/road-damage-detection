import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from road_damage import data, duplicates


def frame(name: str) -> int:
    return int(Path(name).stem.rsplit("_", 1)[1])


def group(name: str) -> int:
    # conftest images: 10 in a row share the same grey level, and it repeats every 100 images
    return (frame(name) // 10) % 10


def test_nearest_returns_index_and_cosine_of_closest_row():
    ref = np.eye(3, dtype=np.float32)
    query = np.array([[0, 1, 0], [0.6, 0, 0.8]], dtype=np.float32)
    idx, sim = duplicates.nearest(query, ref, chunk=1)
    assert idx.tolist() == [1, 2]
    np.testing.assert_allclose(sim, [1.0, 0.8])


def test_embed_rows_are_unit_norm_and_equal_for_equal_images(rdd_root):
    images = sorted((rdd_root / "Japan" / "train" / "images").glob("*.jpg"))[:12]
    e = duplicates.embed(images, arch="resnet18", weights=None, size=32, batch=5, workers=0, device="cpu")
    assert e.shape == (12, 512)
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1, rtol=1e-5)
    np.testing.assert_allclose(e[0], e[9], atol=1e-6)


def test_find_duplicates_gives_nearest_train_image(prepared_cfg, rdd_root):
    out = rdd_root / "yolo"
    result = json.loads((out / "duplicates.json").read_text())["countries"]
    cache = np.load(rdd_root / "embeddings_resnet50.npz")
    assert set(cache.files) == {"Japan", "Japan_names", "Norway", "Norway_names"}
    splits = json.loads((out / "splits.json").read_text())["splits"]
    for country in ("Japan", "Norway"):
        train_groups = {group(n) for n in splits[country]["train"]}
        assert [row[0] for row in result[country]["test"]] == splits[country]["test"]
        for name, nearest_train, sim in result[country]["test"]:
            assert (sim == 1.0) == (group(name) in train_groups)
            if sim == 1.0:
                assert group(nearest_train) == group(name)
    assert "| Japan | 30 |" in duplicates.duplicates_table(result)


def test_subset_yaml_keeps_the_images_in_the_similarity_range(prepared_cfg, rdd_root, tmp_path):
    result = json.loads((rdd_root / "yolo" / "duplicates.json").read_text())["countries"]
    for lo, hi in [(0.5, 1.01), (0.0, 0.5)]:
        expected = sorted(n for c in ("Japan", "Norway") for n, _, s in result[c]["test"] if lo <= s < hi)
        sub = duplicates.subset_yaml(prepared_cfg, "all", "test", (lo, hi), tmp_path / f"sub_{lo}")
        if not expected:
            assert sub is None
            continue
        path, n = sub
        dataset = yaml.safe_load(path.read_text())
        kept = sorted(Path(p).name for f in dataset["test"] for p in Path(f).read_text().split())
        assert kept == expected
        assert n == len(kept)
        assert dataset["train"] == ["Japan_train.txt", "Norway_train.txt"]
    # the 6 grey levels of Norway are all different, no test image has a copy in train
    assert duplicates.subset_yaml(prepared_cfg, "Norway", "test", (0.5, 1.01), tmp_path / "none") is None


def test_embeddings_come_from_the_cache_once_computed(prepared_cfg, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("embed should not run again")

    monkeypatch.setattr(duplicates, "embed", fail)
    names, e = duplicates.embeddings(prepared_cfg)["Japan"]
    assert names == sorted(names) and len(names) == 120
    assert e.shape == (120, 10)


def test_components_follow_chains_of_similar_rows():
    e = np.array([[1, 0, 0], [0.99, 0.141, 0], [0.96, 0.28, 0], [0, 0, 1], [0, 0.6, 0.8]], dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    groups = duplicates.components(e, 0.98, chunk=2)
    # 0-1 and 1-2 are above 0.98, 0-2 is not but they are linked through 1
    assert groups[0] == groups[1] == groups[2]
    assert len({groups[0], groups[3], groups[4]}) == 3


def test_assign_groups_keeps_groups_whole_and_follows_the_ratios():
    groups = np.repeat(np.arange(40), np.arange(40) % 5 + 1)  # 40 groups of 1 to 5 rows
    parts = duplicates.assign_groups(groups, [0.7, 0.1, 0.2], "42-x")
    assert sorted(np.concatenate(list(parts.values()))) == list(range(len(groups)))
    owner = {g: s for s, idx in parts.items() for g in groups[idx]}
    assert all(sum(g in groups[idx] for idx in parts.values()) == 1 for g in owner)
    for split, ratio in zip(data.SPLITS, [0.7, 0.1, 0.2], strict=True):
        assert abs(len(parts[split]) - ratio * len(groups)) <= 5
    again = duplicates.assign_groups(groups, [0.7, 0.1, 0.2], "42-x")
    assert all(np.array_equal(parts[s], again[s]) for s in data.SPLITS)


def test_group_split_keeps_copies_in_one_split(prepared_cfg, rdd_root):
    prepared_cfg["data"].update({"split_dir": "yolo_groups", "group_threshold": 0.5, "train_only": ["Norway"]})
    report = duplicates.group_split(prepared_cfg)
    assert report["Japan"]["groups"] == 10  # the 10 grey levels
    assert report["Norway"] == {"images": 60, "train_only": True}

    out = rdd_root / "yolo_groups"
    splits = json.loads((out / "splits.json").read_text())["splits"]
    seen = {}
    for split, names in splits["Japan"].items():
        for n in names:
            assert seen.setdefault(group(n), split) == split
    assert len(splits["Norway"]["train"]) == 60 and not splits["Norway"]["test"]
    dataset = yaml.safe_load((out / "rdd2022.yaml").read_text())
    assert dataset["train"] == ["Japan_train.txt", "Norway_train.txt"]
    assert dataset["test"] == ["Japan_test.txt"]
    assert data.dataset_yaml(prepared_cfg, "Norway").parent == out

    # no test image has a copy in train any more
    result = duplicates.find_duplicates(prepared_cfg)
    assert all(s < 0.5 for _, _, s in result["Japan"]["test"])
    assert result["Norway"]["test"] == []
    assert "Norway" not in duplicates.duplicates_table(result)
    assert "| Norway | 60 | - | - | 60 | 0 | 0 |" in duplicates.groups_table(report)


def test_group_split_refuses_to_overwrite_the_random_split(prepared_cfg):
    with pytest.raises(ValueError, match="yolo/"):
        duplicates.group_split(prepared_cfg)
