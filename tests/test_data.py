import json
import zipfile
from pathlib import Path

import pytest
import yaml

from road_damage import data
from road_damage.data import Box


def test_parse_voc_reads_every_label_and_float_coordinates(tmp_path):
    path = tmp_path / "a.xml"
    path.write_text(
        "<annotation><size><width>100</width><height>50</height></size>"
        "<object><name> D00 </name><bndbox><xmin>1.0</xmin><ymin>2</ymin><xmax>30.5</xmax><ymax>40</ymax></bndbox>"
        "</object><object><name>Repair</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>5</xmax><ymax>5</ymax>"
        "</bndbox></object><object><name>D10</name></object></annotation>",
        encoding="utf-8",
    )
    assert data.parse_voc(path) == [Box("D00", 1.0, 2.0, 30.5, 40.0), Box("Repair", 0, 0, 5, 5)]


def test_to_yolo_normalises_clips_and_counts_dropped():
    boxes = [
        Box("D00", 0, 0, 50, 25),
        Box("D40", 80, 10, 120, 30),  # past the right edge, clipped to 80..100
        Box("D43", 0, 0, 10, 10),  # not one of the 4 classes
        Box("D10", 10, 10, 10.5, 20),  # less than a pixel wide
    ]
    lines, dropped = data.to_yolo(boxes, 100, 50, data.CLASSES)
    assert lines == ["0 0.250000 0.250000 0.500000 0.500000", "3 0.900000 0.400000 0.200000 0.400000"]
    assert dropped == {"label:D43": 1, "empty_box": 1}


def test_block_split_keeps_blocks_together_and_is_deterministic():
    names = [f"img_{i:04d}" for i in range(1000)]
    a = data.block_split(names, [0.7, 0.1, 0.2], 50, "42-Japan")
    assert a == data.block_split(list(reversed(names)), [0.7, 0.1, 0.2], 50, "42-Japan")
    assert a != data.block_split(names, [0.7, 0.1, 0.2], 50, "42-Norway")
    assert sorted(sum(a.values(), [])) == names
    assert [len(a[s]) for s in data.SPLITS] == [700, 100, 200]
    blocks = {s: {int(n[4:]) // 50 for n in a[s]} for s in data.SPLITS}
    assert sum(len(b) for b in blocks.values()) == 20  # no block is shared by two splits


@pytest.mark.parametrize("ratios", [[0.7, 0.1], [0.8, 0.3, 0.1], [1.2, -0.1, -0.1]])
def test_block_split_rejects_bad_ratios(ratios):
    with pytest.raises(ValueError):
        data.block_split(["a", "b"], ratios, 1, "s")


def test_index_country_pairs_images_and_xmls(rdd_root):
    items = data.index_country(rdd_root, "Japan")
    assert len(items) == 120
    assert items[2][1] is None
    assert items[3][1].name == "Japan_000003.xml"
    with pytest.raises(FileNotFoundError):
        data.index_country(rdd_root, "India")


def test_prepare_writes_labels_splits_and_dataset_files(rdd_cfg, rdd_root):
    stats = data.prepare(rdd_cfg)
    labels = rdd_root / "Japan" / "train" / "labels"
    assert len(list(labels.glob("*.txt"))) == 120
    assert (labels / "Japan_000000.txt").read_text() == ""
    assert len((labels / "Japan_000001.txt").read_text().splitlines()) == 2

    s = stats["Japan"]
    assert s["images"] == 120
    assert s["no_xml"] == 1
    assert s["background"] == 41  # every 3rd image, plus the one without xml
    assert s["boxes"] == {"D00": 79, "D10": 0, "D20": 0, "D40": 1}
    assert s["dropped"] == {"label:D43": 1}
    assert s["sizes"] == {"64x48": 120}

    out = rdd_root / "yolo"
    train = (out / "Japan_train.txt").read_text().splitlines()
    assert len(train) == 80  # 8 of the 12 blocks of 10
    assert all(Path(p).is_absolute() and Path(p).exists() for p in train)
    dataset = yaml.safe_load((out / "rdd2022.yaml").read_text())
    assert dataset["names"] == {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
    assert dataset["test"] == ["Japan_test.txt", "Norway_test.txt"]
    assert yaml.safe_load((out / "rdd2022_Norway.yaml").read_text())["train"] == ["Norway_train.txt"]
    splits = json.loads((out / "splits.json").read_text())
    assert sorted(sum(splits["splits"]["Norway"].values(), [])) == [f"Norway_{i:06d}.jpg" for i in range(60)]


def test_dataset_yaml_needs_prepare(rdd_cfg):
    with pytest.raises(FileNotFoundError):
        data.dataset_yaml(rdd_cfg)


def test_stats_table(rdd_cfg):
    table = data.stats_table(data.prepare(rdd_cfg), data.CLASSES)
    assert "| Japan | 120 | 41 | 79 | 0 | 0 | 1 | 1 | 64x48 |" in table


def test_sequence_check_detects_frames_of_one_drive(rdd_root):
    images = sorted((rdd_root / "Japan" / "train" / "images").glob("*.jpg"))
    r = data.sequence_check(images, pairs=100)
    assert r["ratio"] < 0.3


def _zip_tree(zip_path, files):
    with zipfile.ZipFile(zip_path, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)


@pytest.mark.parametrize("nested", [False, True])
def test_extract_puts_country_folders_under_root(tmp_path, nested):
    files = {"RDD2022/Japan/train/images/Japan_000000.jpg": b"x", "RDD2022/Japan/train/annotations/xmls/a.xml": b"x"}
    archive = tmp_path / data.RDD_ARCHIVE
    if nested:
        _zip_tree(tmp_path / "Japan.zip", {k.removeprefix("RDD2022/"): v for k, v in files.items()})
        _zip_tree(archive, {"RDD2022/Japan.zip": (tmp_path / "Japan.zip").read_bytes()})
    else:
        _zip_tree(archive, files)
    root = tmp_path / "datasets" / "RDD2022"
    data.extract(archive, root)
    assert (root / "Japan" / "train" / "images" / "Japan_000000.jpg").exists()
    assert not (root.parent / "RDD2022_extract").exists()


def test_download_checks_md5_of_existing_archive(tmp_path):
    root = tmp_path / "RDD2022"
    archive = tmp_path / data.RDD_ARCHIVE
    _zip_tree(archive, {"RDD2022/Norway/train/images/Norway_000000.jpg": b"x"})
    with pytest.raises(ValueError, match="MD5 mismatch"):
        data.download(root, url="http://unused", md5="0" * 32)
    data.download(root, url="http://unused", md5=data.md5sum(archive))
    assert (root / "Norway" / "train" / "images" / "Norway_000000.jpg").exists()
