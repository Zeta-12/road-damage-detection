# Road Damage Detection

[![CI](https://github.com/Zeta-12/road-damage-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Zeta-12/road-damage-detection/actions/workflows/ci.yml)

Detection of cracks and potholes in images taken from vehicles, on the RDD2022 dataset (Japan, India, Norway, United
States, Czech Republic, China). I compare a CNN detector (YOLO) with a transformer detector (RT-DETR), measure how a
model trained in some countries works in the others, and deploy the best one on an edge device (OpenVINO, INT8, C++).

In short:

- YOLO26s and RT-DETR-L get the same test mAP50 (0.600 and 0.601). RT-DETR-L needs 5x the compute and is only
  clearly better in Norway.
- The score goes from 0.29 to 0.93 depending on the subset. The 0.93 of China_MotorBike is inflated by
  near-duplicates between train and test (the same damage photographed several times), found with a nearest
  neighbour search on image embeddings. On a split that keeps near-duplicates together, YOLO26s gets 0.544.
- Norway is limited by the input size: +7 points of mAP50 by running YOLO26s at 1280 px instead of 640.
- On an Intel Core Ultra laptop, INT8 on the iGPU runs at 9 ms per image (111 img/s) for 0.9 point of mAP50. The NPU
  should stay in FP16: INT8 costs 2.7 points there and is not faster.

Status: training, near-duplicate analysis and OpenVINO deployment done, results below. Next: Norway at 1280 px,
cross-country, C++ app.

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
possible, they are measured in [Results](#results).

## Results

### YOLO26s on all countries

YOLO26s fine-tuned from the COCO weights on the 7 subsets together: 640 px, batch 16, 100 epochs, 8.2 h on an
RTX 3060 Ti (8 GB). With `optimizer=auto` Ultralytics picks MuSGD at lr 0.01. `best.pt` is epoch 67, the best val
mAP50-95 (0.314, mAP50 0.606).

Test split (`rdd evaluate`):

| Test set | Images | mAP50 | mAP50-95 | Recall | AP50 D00 | AP50 D10 | AP50 D20 | AP50 D40 |
|---|---|---|---|---|---|---|---|---|
| all | 7676 | 0.600 | 0.311 | 0.556 | 0.603 | 0.580 | 0.710 | 0.508 |
| Japan | 2101 | 0.579 | 0.278 | 0.562 | 0.509 | 0.489 | 0.732 | 0.586 |
| India | 1541 | 0.401 | 0.190 | 0.394 | 0.384 | 0.097 | 0.678 | 0.445 |
| Norway | 1632 | 0.291 | 0.125 | 0.299 | 0.382 | 0.259 | 0.333 | 0.191 |
| United_States | 961 | 0.653 | 0.384 | 0.623 | 0.792 | 0.677 | 0.741 | 0.403 |
| Czech | 566 | 0.410 | 0.160 | 0.452 | 0.466 | 0.334 | 0.524 | 0.314 |
| China_MotorBike | 395 | 0.927 | 0.580 | 0.872 | 0.915 | 0.866 | 0.943 | 0.985 |
| China_Drone | 480 | 0.701 | 0.398 | 0.687 | 0.768 | 0.812 | 0.398 | 0.826 |

1. **The subset matters much more than the class: 0.29 mAP50 in Norway, 0.93 in China_MotorBike.** Points 2 and 3
   explain part of it. Some AP50 values rest on very few boxes: India has 13 D10 boxes in test, the United States
   24 D40 boxes.

2. **There are near-duplicates between train and test, a lot in China_MotorBike.** `rdd duplicates` embeds every
   image with an ImageNet ResNet-50 and finds the nearest train image of each val / test image (cosine similarity).
   Looking at the pairs, most of them are the same place from 0.96 up, in every subset. In China_MotorBike the same
   damage was photographed several times and the copies go further down, the 0.879 pair below is the same crack with
   the same bicycle at the top:

   ![China_MotorBike test images and their nearest train image](docs/assets/china_motorbike_pairs.jpg)

   mAP50 on the test images with and without a near copy in train (`rdd evaluate --similarity 0.96 1.01`, then
   `--similarity 0 0.96`):

   | Test set | Images >= 0.96 | mAP50 | Images < 0.96 | mAP50 |
   |---|---|---|---|---|
   | all | 1066 | 0.735 | 6610 | 0.579 |
   | Japan | 30 | 0.818 | 2071 | 0.575 |
   | India | 45 | 0.499 | 1496 | 0.401 |
   | Norway | 589 | 0.249 | 1043 | 0.315 |
   | United_States | 56 | 0.775 | 905 | 0.649 |
   | Czech | 13 | 0.497 | 553 | 0.410 |
   | China_MotorBike | 300 | 0.931 | 95 | 0.891 |
   | China_Drone | 33 | 0.960 | 447 | 0.681 |

   - Without them the overall mAP50 goes from 0.600 to 0.579. Japan, India, the United States and Czech hardly move.
   - 300 of the 395 China_MotorBike test images are above 0.96 and the 95 others still include copies. Its 0.927
     says more about memorisation than about detection, I don't use it to compare models.
   - Norway goes the other way (0.249 with a near copy, 0.315 without). Many of its near copies are the same road a
     few metres further, but at 640 px the cracks are too small for the copy to help (point 3).

   A test set without leaks needs a split by groups of near-duplicates, see
   [below](#yolo26s-on-a-split-without-near-duplicates).

3. **Norway is limited by the input size.** Its images are 3643 to 4040 px wide and 640 px divides them by 6. Same
   weights, Norway test only, larger inference size (`rdd evaluate --countries Norway --imgsz 1280`):

   | Inference size | 640 | 1280 | 1920 |
   |---|---|---|---|
   | Norway mAP50 | 0.291 | 0.364 | 0.325 |
   | Norway recall | 0.299 | 0.379 | 0.350 |

   1920 loses part of the gain, probably because the objects are then far from the sizes seen in training at 640.
   Next step: train at 1280, or on tiles.

4. **Training longer did not help.** The val losses stop going down after epoch 50 to 70, val mAP50 peaks at 0.607
   around epoch 70 and ends at 0.593 at epoch 100. Turning mosaic off for the last 10 epochs (Ultralytics default)
   made the train loss drop and the val mAP go down further. The next runs use fewer epochs.

5. **Classes are rarely confused with each other.** In the confusion matrix of the test set, at most 7 % of the boxes
   of a class are predicted as another class (D20 as D00). The errors are missed damage and detections on intact road,
   half of these false detections are D00.

### RT-DETR-L vs YOLO26s

RT-DETR-L fine-tuned from the COCO weights on the same data, with the usual RT-DETR recipe (`configs/rtdetr.yaml`):
AdamW, lr 1e-4, batch 8 (16 does not fit in 8 GB), 40 epochs, 13.1 h. Ultralytics accumulates the gradients up to
64 images per step for both models. `best.pt` is epoch 18, the best val mAP50-95 (0.298, mAP50 0.613).

| | YOLO26s | RT-DETR-L |
|---|---|---|
| Parameters | 9.5 M | 32.0 M |
| GFLOPs at 640 px | 20.8 | 105.3 |
| GPU inference per image (Ultralytics val, FP16, batch 16 / 8) | 4.0 ms | 16.2 ms |
| Training | 8.2 h, best epoch 67 | 13.1 h, best epoch 18 |
| Test mAP50 | 0.600 | 0.601 |
| Test mAP50-95 | 0.311 | 0.293 |

Test mAP50 per subset:

| Test set | YOLO26s | RT-DETR-L | Difference |
|---|---|---|---|
| all | 0.600 | 0.601 | +0.000 |
| Japan | 0.579 | 0.578 | -0.002 |
| India | 0.401 | 0.367 | -0.033 |
| Norway | 0.291 | 0.388 | +0.097 |
| United_States | 0.653 | 0.623 | -0.031 |
| Czech | 0.410 | 0.362 | -0.047 |
| China_MotorBike | 0.927 | 0.906 | -0.021 |
| China_Drone | 0.701 | 0.641 | -0.060 |

1. **Same mAP50 overall for 5x the compute.** The RT-DETR-L boxes are less tight (mAP50-95 0.293 vs 0.311) and it is
   lower in every subset except Japan (same) and Norway. For the edge part, the model to deploy is YOLO26s.

2. **Norway is the exception: 0.388 vs 0.291.** RT-DETR-L is better on the 4 classes there (D10 0.402 vs 0.259),
   with a recall of 0.405 vs 0.299, and it beats YOLO26s run at 1280 px (0.364). Run at 1280 px itself, it drops to
   0.353: unlike YOLO26s it does not gain from a bigger input without retraining. I haven't checked why it is better
   there. At 640 px the Norway cracks are long and less than a pixel wide; my guess is that the self-attention of the
   RT-DETR encoder, which sees the whole image at stride 32, helps to follow them.

3. **It learns faster and overfits sooner.** Val mAP50 at epoch 5: 0.539 (YOLO26s: 0.330). It peaks at 0.616 at
   epoch 24, ends at 0.603 at epoch 40, and the val GIoU loss goes up from epoch 15. 20 epochs (6.6 h) would have
   been enough.

### YOLO26s on a split without near-duplicates

`rdd group-split --config configs/grouped.yaml` links the images whose embeddings have a cosine similarity of at
least 0.96 (the threshold found on the pairs above), takes the connected components as groups and gives whole groups
to train, val and test: 70 / 10 / 20 per subset, biggest groups first, each one to the split furthest below its
target. After it, the closest train image of any test image is at 0.960.

- China_MotorBike goes to train only. Its copies go down to 0.87, and at 0.90 one group already holds 79 % of its
  images, so there is no clean split for it.
- Norway chains: one group holds 2665 of its images (33 %), look-alike rural roads. It goes to train and the Norway
  test is drawn from the rest.

YOLO26s retrained on this split (`rdd train --config configs/grouped.yaml`: 70 epochs, 5.8 h, best val epoch 59):

| Test set | Random split | Split by groups | Difference |
|---|---|---|---|
| all | 0.600 | 0.544 | -0.056 |
| Japan | 0.579 | 0.579 | 0.000 |
| India | 0.401 | 0.406 | +0.005 |
| Norway | 0.291 | 0.251 | -0.040 |
| United_States | 0.653 | 0.638 | -0.015 |
| Czech | 0.410 | 0.387 | -0.023 |
| China_Drone | 0.701 | 0.697 | -0.004 |

The test images are not the same in the two splits, so a difference per subset mixes the effect of the leaks and a
change of test images.

1. **0.544 mAP50 (0.272 mAP50-95) is the number to quote**, on the 6 subsets that can be split cleanly. Most of the
   drop of "all" comes from China_MotorBike leaving the test set.
2. **Per subset the leaks were worth 0 to 2 points**, as the similarity filter above suggested, except in Norway
   (-4 points). Its test no longer shares any road with train; I haven't looked further.
3. Norway at 1280 px with this model: 0.328, against 0.251 at 640.

### Deployment on an Intel laptop (OpenVINO)

The YOLO26s of the split without near-duplicates, exported with `rdd export` (OpenVINO 2026.4): FP32, FP16 (weights
stored in FP16) and INT8 (NNCF post-training quantization, calibrated on 50 train images per subset, Ultralytics
keeps the detection head in floating point). Laptop: Intel Core Ultra 5 235U with its CPU, Intel Graphics iGPU and
AI Boost NPU, plugged in, Balanced power mode. Batch 1, 640x640 input.

Latency is the median of 200 inferences one image at a time (LATENCY hint), throughput uses parallel requests
(THROUGHPUT hint), `rdd benchmark`:

| | CPU | iGPU | NPU |
|---|---|---|---|
| PyTorch FP32 | 125 ms | | |
| OpenVINO FP32 | 71.5 ms, 18 img/s | 16.5 ms, 47 img/s | 14.7 ms, 80 img/s |
| OpenVINO FP16 | 80.9 ms, 16 img/s | 17.0 ms, 47 img/s | 14.9 ms, 79 img/s |
| OpenVINO INT8 | 29.9 ms, 48 img/s | 9.0 ms, 111 img/s | 14.6 ms, 85 img/s |

Test mAP50 on 640x640 inputs (`rdd evaluate --device intel:cpu`, `intel:gpu` or `intel:npu`):

| | CPU | iGPU | NPU |
|---|---|---|---|
| PyTorch FP32 | 0.525 | | |
| OpenVINO FP32 / FP16 | 0.525 | 0.525 | 0.525 |
| OpenVINO INT8 | 0.517 | 0.516 | 0.498 |

1. **The export gives the same boxes as PyTorch, the evaluation protocol is what differs.** Ultralytics evaluates a
   PyTorch model on rectangular batches, with a border of half a stride (16 px) around the image, while a model
   exported with a fixed size gets exactly 640x640. That alone moves the mAP50 from 0.544 to 0.525 (China_Drone: 0.697
   to 0.652). With `rdd evaluate --square` PyTorch gets the same inputs as the export and the same mAP, 0.525 overall
   and in every subset to the third decimal.
2. **FP16 costs nothing**, on the 3 devices. On the iGPU and the NPU, FP32 and FP16 also run at the same speed: both
   compute in FP16 by default.
3. **INT8 costs 0.8 to 0.9 point on the CPU and the iGPU.** It is 2.4x faster than FP32 on the CPU (4.2x faster than
   PyTorch) and 1.8x on the iGPU. The fastest setup is INT8 on the iGPU: 9 ms per image, 111 img/s.
4. **The same INT8 model loses 2.7 points on the NPU** (India -4.7, Norway -5.2) and is not faster there than FP16
   (14.6 vs 14.9 ms), so the NPU should run the FP16 model. I haven't looked at what the NPU compiler does
   differently with the INT8 graph.
5. The CPU timings moved by about 20 % between two runs on different days (PyTorch 158 then 125 ms), the NPU ones did
   not. The ratios between precisions stayed the same.

## Plan

- [x] YOLO26s on all countries, mAP per country
- [x] RT-DETR-L on all countries
- [x] Near-duplicates between train and test
- [x] Split by groups of near-duplicates and retrain, for a test set without leaks
- [ ] Norway at 1280 px (training at that size, or tiles)
- [ ] Cross-country: train on one subset, test on the others
- [x] OpenVINO FP16 / INT8 (post-training quantization), latency and mAP on CPU / iGPU / NPU. Quantization-aware
  training is not needed for the CPU and the iGPU (-0.8 to -0.9 point)
- [ ] INT8 on the NPU: find where the 2.7 points go (layer by layer against the FP16 run)
- [ ] Fixed 672x672 input with the image at 640, the border of the rectangular batches, to get the 2 points back
- [ ] C++ inference app with OpenVINO

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
uv run rdd duplicates         # nearest train image of every val / test image
uv run rdd evaluate --weights runs/yolo26s_all/weights/best.pt --similarity 0 0.96
uv run rdd train --config configs/rtdetr.yaml

# split without near-duplicates (reuses the cached embeddings of rdd duplicates)
uv run rdd group-split --config configs/grouped.yaml
uv run rdd train --config configs/grouped.yaml --name yolo26s_groups
uv run rdd evaluate --config configs/grouped.yaml --weights runs/yolo26s_groups/weights/best.pt
```

OpenVINO export and benchmark on an Intel CPU / iGPU / NPU (`--extra openvino`):

```bash
uv sync --extra openvino
W=runs/yolo26s_groups/weights
uv run rdd export --config configs/grouped.yaml --weights $W/best.pt    # FP32, FP16, INT8
uv run rdd benchmark --config configs/grouped.yaml --weights $W/best.pt \
    --models $W/best_fp32_openvino_model $W/best_fp16_openvino_model $W/best_int8_openvino_model
uv run rdd evaluate --config configs/grouped.yaml --weights $W/best_int8_openvino_model \
    --device intel:npu --tag int8_npu
uv run rdd evaluate --config configs/grouped.yaml --weights $W/best.pt --square    # same inputs as the export
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
  data.py        download, VOC to YOLO labels, block split per country, dataset files, stats
  train.py       Ultralytics training, mAP per country
  duplicates.py  nearest train image of every val / test image (ResNet-50 embeddings), split by groups
  deploy.py      OpenVINO export (FP32, FP16, INT8), latency and throughput on CPU / iGPU / NPU
  config.py      defaults and YAML overrides
  cli.py         `rdd` command
tests/           unit tests on a small synthetic dataset in RDD2022 layout, 1-epoch end-to-end runs (Ultralytics,
                 OpenVINO export)
configs/         default.yaml with all the parameters, rtdetr.yaml, grouped.yaml
```

## Dataset and licence

[RDD2022](https://doi.org/10.6084/m9.figshare.21431547) (Arya et al., [arXiv:2209.08538](https://arxiv.org/abs/2209.08538))
is distributed under CC BY 4.0 and is not included in this repository. `rdd download` fetches it from figshare and
checks the MD5. The 8 photos in `docs/assets/china_motorbike_pairs.jpg` come from RDD2022.

Code: AGPL-3.0, because it uses [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0).

## Author

**Julien Beloin**, Machine Learning & Computer Vision Engineer
([LinkedIn](https://www.linkedin.com/in/j-beloin), [GitHub](https://github.com/Zeta-12))
