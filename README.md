# Road Damage Detection

[![CI](https://github.com/Zeta-12/road-damage-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Zeta-12/road-damage-detection/actions/workflows/ci.yml)

Detection of cracks and potholes in images taken from vehicles, on the RDD2022 dataset (Japan, India, Norway, United
States, Czech Republic, China). I compare a CNN detector (YOLO) with a transformer detector (RT-DETR), measure how a
model trained in some countries works in the others, and deploy the best one on an edge device (OpenVINO, INT8, C++).

Status: data pipeline done, first training running (YOLO26s on all countries).

## Dataset

| Class | Damage |
|---|---|
| D00 | longitudinal crack |
| D10 | transverse crack |
| D20 | alligator crack |
| D40 | pothole |

Labelled images and boxes kept (output of `rdd prepare`):

| Subset | Images | No damage | D00 | D10 | D20 | D40 | Image size |
|---|---|---|---|---|---|---|---|
| Japan | 10506 | 2606 | 4049 | 3979 | 6198 | 2243 | 600x600 (97 %) |
| India | 7706 | 4483 | 1555 | 68 | 2021 | 3187 | 720x720 |
| Norway | 8161 | 5247 | 8570 | 1730 | 468 | 461 | 3643x2041 to 4040x2035 |
| United_States | 4805 | 0 | 6750 | 3295 | 834 | 135 | 640x640 |
| Czech | 2829 | 1757 | 988 | 399 | 161 | 197 | 600x600 |
| China_MotorBike | 1977 | 43 | 2678 | 1096 | 641 | 235 | 512x512 |
| China_Drone | 2401 | 482 | 1426 | 1263 | 293 | 86 | 512x512 |
| **total** | 38385 | 14618 | 26016 | 11830 | 10616 | 6544 | |

- As in the challenge, only the 4 classes are kept. The 10705 boxes with other labels are dropped, mostly in Japan:
  blurred crosswalks and white lines (D43, D44), manholes (D50), repairs and a few rare classes.
- The subsets are very different: no image without damage in the United States, few transverse cracks in India,
  drone views for China_Drone. Norway images are 6x wider than the others: resized to 640 px for training, a crack
  5 px wide becomes less than 1 px.

The images of the official challenge test set have no labels, so I split the labelled `train` folders of each
subset into train / val / test (70 / 10 / 20, seed 42). I first wrote a split by blocks of consecutive images, in
case the numbering followed the capture order: consecutive frames of a drive are near-duplicates and would leak
between train and test. `rdd stats --sequences` compares 32x32 thumbnails of consecutive images and of random pairs.
The ratio of the mean differences is between 0.96 and 1.04 in every subset, so the numbering is shuffled and the
split is random per image (`block_size: 1`). Near-duplicates that are not neighbours in the numbering are still
possible, checking that needs a nearest neighbour search (see [Plan](#plan)).

## Plan

1. YOLO26 and RT-DETR fine-tuned on all countries, mAP per country.
2. Cross-country: train on one country, test on the others.
3. Near-duplicates between train and test (nearest neighbours on image embeddings).
4. OpenVINO FP16 / INT8 (post-training quantization, then quantization-aware training if needed), latency and mAP
   on CPU / iGPU / NPU.
5. C++ inference app with OpenVINO.

## Quickstart

Requires Python 3.11 to 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Zeta-12/road-damage-detection.git
cd road-damage-detection
uv sync                       # CPU wheels

uv run rdd download           # RDD2022, 13.3 GB, CC BY 4.0, into datasets/RDD2022
uv run rdd prepare            # YOLO labels, split, dataset files in datasets/RDD2022/yolo/
uv run rdd stats --sequences  # images and boxes per country, consecutive image check
uv run rdd train --model yolo26s.pt
uv run rdd evaluate --weights runs/yolo26s_all/weights/best.pt
```

On an NVIDIA GPU, replace the CPU wheels with the CUDA build that matches your driver, then use `--no-sync` so uv
does not put the CPU wheels back:

```bash
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu134
uv run --no-sync rdd train --model yolo26s.pt
```

## Repository layout

```
src/road_damage/
  data.py      download, VOC to YOLO labels, block split per country, dataset files, stats
  train.py     Ultralytics training, mAP per country
  config.py    defaults and YAML overrides
  cli.py       `rdd` command
tests/         unit tests on a small synthetic dataset in RDD2022 layout, and a 1-epoch end-to-end run
configs/       default.yaml with all the parameters
```

## Dataset and licence

[RDD2022](https://doi.org/10.6084/m9.figshare.21431547) (Arya et al., [arXiv:2209.08538](https://arxiv.org/abs/2209.08538))
is distributed under CC BY 4.0 and is not included in this repository. `rdd download` fetches it from figshare and
checks the MD5.

Code: AGPL-3.0, because it uses [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0).

## Author

**Julien Beloin**, Machine Learning & Computer Vision Engineer
([LinkedIn](https://www.linkedin.com/in/j-beloin), [GitHub](https://github.com/Zeta-12))
