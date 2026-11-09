"""`rdd` command line."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from road_damage.config import load_config


def _overrides(args: argparse.Namespace) -> dict:
    data, train = {}, {}
    if getattr(args, "data_root", None):
        data["root"] = args.data_root
    if getattr(args, "countries", None):
        data["countries"] = args.countries
    for key in ("model", "epochs", "batch", "imgsz", "device", "workers"):
        if getattr(args, key, None) is not None:
            train[key] = getattr(args, key)
    out = {"data": data, "train": train}
    if getattr(args, "output_dir", None):
        out["output_dir"] = args.output_dir
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rdd", description="Road damage detection on RDD2022.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=None, help="YAML config (optional).")
    common.add_argument("--data-root", help="Folder containing the RDD2022 country folders.")
    common.add_argument("--countries", nargs="+", help="Subset of countries (default: all 7).")
    common.add_argument("--output-dir", help="Where runs are written (default: runs/).")

    sub.add_parser("download", parents=[common], help="Download and extract RDD2022 (13.3 GB, CC BY 4.0).")
    sub.add_parser("prepare", parents=[common], help="YOLO labels, train / val / test split, dataset files.")
    p_stats = sub.add_parser("stats", parents=[common], help="Images and boxes per country (after prepare).")
    p_stats.add_argument("--sequences", action="store_true", help="Also check if consecutive images look alike.")
    p_train = sub.add_parser("train", parents=[common], help="Fine-tune a detector.")
    p_train.add_argument("--model", help="Ultralytics weights, e.g. yolo26s.pt or rtdetr-l.pt.")
    p_train.add_argument("--country", help="Train on one country only (default: all).")
    p_train.add_argument("--name", help="Run name (default: <model>_<country>).")
    p_train.add_argument("--epochs", type=int)
    p_train.add_argument("--batch", type=int)
    p_train.add_argument("--imgsz", type=int)
    p_train.add_argument("--workers", type=int)
    p_train.add_argument("--device", help="0, cpu, ... (default: Ultralytics choice).")
    p_eval = sub.add_parser("evaluate", parents=[common], help="mAP per country on the test split.")
    p_eval.add_argument("--weights", required=True, help="best.pt of a run.")
    p_eval.add_argument("--split", default="test", choices=["val", "test"])
    p_eval.add_argument("--batch", type=int)
    p_eval.add_argument("--device")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config, _overrides(args))
    root = Path(cfg["data"]["root"])

    if args.command == "download":
        from road_damage.data import download

        download(root)
    elif args.command == "prepare":
        from road_damage.data import prepare, stats_table

        print(stats_table(prepare(cfg), cfg["data"]["classes"]))
    elif args.command == "stats":
        from road_damage.data import index_country, sequence_check, stats_table

        stats = json.loads((root / "yolo" / "stats.json").read_text(encoding="utf-8"))
        print(stats_table(stats, cfg["data"]["classes"]))
        if args.sequences:
            for country in cfg["data"]["countries"]:
                r = sequence_check([p for p, _ in index_country(root, country)], seed=cfg["data"]["seed"])
                c, rnd, ratio = r["consecutive"], r["random"], r["ratio"]
                print(f"{country:16s} consecutive {c:6.1f}  random {rnd:6.1f}  ratio {ratio:.2f}")
    elif args.command == "train":
        from road_damage.train import train

        print(train(cfg, country=args.country, name=args.name))
    elif args.command == "evaluate":
        from road_damage.train import evaluate, markdown_table

        print(markdown_table(evaluate(cfg, args.weights, args.split), cfg["data"]["classes"]))


if __name__ == "__main__":
    main()
