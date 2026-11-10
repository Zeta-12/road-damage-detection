import json
from pathlib import Path

import numpy as np
import yaml

from road_damage import duplicates


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
    assert set(np.load(out / "embeddings_resnet50.npz").files) == {"Japan", "Japan_names", "Norway", "Norway_names"}
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
