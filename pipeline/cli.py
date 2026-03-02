import argparse
import json
import os

from pipeline.config import AudioConfig, TrainConfig

DEFAULT_POSITIVE_EVENTS = [
    "asking help",
    "screaming",
    "crying sobbing wail",
    "whispering",
    "cough",
]


def _resolve_dataset_root(dataset_root: str) -> str:
    if os.path.isdir(dataset_root):
        return dataset_root
    nested = os.path.join(dataset_root, "DisasterDataset")
    if os.path.isdir(nested):
        return nested
    return dataset_root


def _dependency_hint(exc: ModuleNotFoundError) -> str:
    pkg = exc.name or "required package"
    return (
        f"Missing dependency '{pkg}'. Install requirements first:\n"
        "python3 -m pip install -r requirements.txt"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal survivor detection pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Build train/val/test manifest from raw dataset.")
    p_prepare.add_argument("--dataset_root", default="data/DisasterDataset")
    p_prepare.add_argument("--out_dir", default="artifacts/data")
    p_prepare.add_argument("--generated_ratio", type=float, default=1.0)
    p_prepare.add_argument("--val_split", type=float, default=0.15)
    p_prepare.add_argument("--test_split", type=float, default=0.10)
    p_prepare.add_argument("--seed", type=int, default=42)
    p_prepare.add_argument("--positive_events", nargs="+", default=DEFAULT_POSITIVE_EVENTS)

    p_train = sub.add_parser("train", help="Train binary distress classifier from a manifest.")
    p_train.add_argument("--manifest", default="artifacts/data/manifest.csv")
    p_train.add_argument("--out_dir", default="artifacts/model")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--distress_weight", type=float, default=2.0)
    p_train.add_argument("--num_workers", type=int, default=2)

    p_infer = sub.add_parser("infer", help="Quick inference on one audio file.")
    p_infer.add_argument("--model", default="artifacts/model/best_model.pth")
    p_infer.add_argument("--audio", required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "prepare":
        try:
            from pipeline.data import build_manifest
        except ModuleNotFoundError as exc:
            raise SystemExit(_dependency_hint(exc)) from exc

        train_cfg = TrainConfig(
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed,
        )
        manifest_path, summary_path = build_manifest(
            dataset_root=_resolve_dataset_root(args.dataset_root),
            output_dir=args.out_dir,
            train_cfg=train_cfg,
            positive_events=args.positive_events,
            generated_ratio=args.generated_ratio,
        )
        print(json.dumps({"manifest": manifest_path, "summary": summary_path}, indent=2))
        return

    if args.command == "train":
        try:
            from pipeline.train import train_model
        except ModuleNotFoundError as exc:
            raise SystemExit(_dependency_hint(exc)) from exc

        train_cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            distress_weight=args.distress_weight,
            num_workers=args.num_workers,
        )
        out = train_model(
            manifest_path=args.manifest,
            output_dir=args.out_dir,
            audio_cfg=AudioConfig(),
            train_cfg=train_cfg,
        )
        print(json.dumps(out, indent=2))
        return

    if args.command == "infer":
        try:
            from pipeline.infer import infer_one
        except ModuleNotFoundError as exc:
            raise SystemExit(_dependency_hint(exc)) from exc

        out = infer_one(model_path=args.model, audio_path=args.audio, audio_cfg=AudioConfig())
        print(json.dumps(out, indent=2))
        return


if __name__ == "__main__":
    main()
