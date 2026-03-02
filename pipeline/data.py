import json
import os
import random
from dataclasses import asdict
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from pipeline.config import AudioConfig, TrainConfig


DEFAULT_POSITIVE_EVENTS = [
    "asking help",
    "screaming",
    "crying sobbing wail",
    "whispering",
    "cough",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _list_wavs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return [
        os.path.join(root, f)
        for f in sorted(os.listdir(root))
        if os.path.isfile(os.path.join(root, f)) and f.lower().endswith(".wav")
    ]


def build_manifest(
    dataset_root: str,
    output_dir: str,
    train_cfg: TrainConfig,
    positive_events: List[str] | None = None,
    generated_ratio: float = 1.0,
) -> Tuple[str, str]:
    seed_everything(train_cfg.seed)
    positive_events = positive_events or DEFAULT_POSITIVE_EVENTS
    positive_set = {x.strip().lower() for x in positive_events}

    reference_root = os.path.join(dataset_root, "reference")
    generated_root = os.path.join(dataset_root, "generated")

    rows: List[Dict[str, str | int]] = []
    for source, source_root in [("reference", reference_root), ("generated", generated_root)]:
        if not os.path.isdir(source_root):
            continue
        for event in sorted(os.listdir(source_root)):
            event_dir = os.path.join(source_root, event)
            if not os.path.isdir(event_dir):
                continue
            label_id = 1 if event.strip().lower() in positive_set else 0
            label_name = "distress" if label_id == 1 else "non_distress"
            for path in _list_wavs(event_dir):
                rows.append(
                    {
                        "path": os.path.abspath(path),
                        "event": event,
                        "source": source,
                        "label_id": label_id,
                        "label_name": label_name,
                    }
                )

    if not rows:
        raise RuntimeError(f"No wav files found in {dataset_root}")

    df = pd.DataFrame(rows)
    ref = df[df["source"] == "reference"].copy()
    gen = df[df["source"] == "generated"].copy()
    if ref.empty:
        raise RuntimeError("No reference data found. Expected reference/ folder with wav files.")

    train_ref, test_ref = train_test_split(
        ref,
        test_size=train_cfg.test_split,
        random_state=train_cfg.seed,
        stratify=ref["label_id"],
    )
    rel_val = train_cfg.val_split / (1.0 - train_cfg.test_split)
    train_ref, val_ref = train_test_split(
        train_ref,
        test_size=rel_val,
        random_state=train_cfg.seed,
        stratify=train_ref["label_id"],
    )

    train_ref["split"] = "train"
    val_ref["split"] = "val"
    test_ref["split"] = "test"

    sampled_gen = []
    if not gen.empty and generated_ratio > 0:
        gen["split"] = "train"
        train_ref_counts = train_ref["label_id"].value_counts().to_dict()
        for label_id, group in gen.groupby("label_id"):
            cap = int(round(train_ref_counts.get(label_id, 0) * generated_ratio))
            cap = min(cap, len(group))
            if cap > 0:
                sampled_gen.append(group.sample(n=cap, random_state=train_cfg.seed))
    gen_train = pd.concat(sampled_gen, ignore_index=True) if sampled_gen else pd.DataFrame(columns=df.columns)

    manifest = pd.concat([train_ref, val_ref, test_ref, gen_train], ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    summary = {
        "total_rows": int(len(manifest)),
        "reference_rows": int(len(ref)),
        "generated_rows": int(len(gen)),
        "train_rows": int((manifest["split"] == "train").sum()),
        "val_rows": int((manifest["split"] == "val").sum()),
        "test_rows": int((manifest["split"] == "test").sum()),
        "positive_events": positive_events,
        "train_config": asdict(train_cfg),
    }
    summary_path = os.path.join(output_dir, "manifest_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return manifest_path, summary_path


class SurvivorDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, split: str, audio_cfg: AudioConfig):
        self.df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        self.audio_cfg = audio_cfg
        if self.df.empty:
            raise RuntimeError(f"No rows found for split '{split}' in manifest.")

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_cfg.sample_rate,
            n_fft=audio_cfg.n_fft,
            hop_length=audio_cfg.hop_length,
            n_mels=audio_cfg.n_mels,
            f_min=audio_cfg.f_min,
            f_max=audio_cfg.f_max,
            power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav, sr = torchaudio.load(row["path"])
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.audio_cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.audio_cfg.sample_rate)

        target_len = self.audio_cfg.num_samples
        cur_len = wav.shape[-1]
        if cur_len < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - cur_len))
        elif cur_len > target_len:
            start = (cur_len - target_len) // 2
            wav = wav[:, start : start + target_len]

        mel = self.db(self.mel(wav))
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel, int(row["label_id"])


def create_dataloaders(
    manifest_path: str,
    audio_cfg: AudioConfig,
    train_cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    df = pd.read_csv(manifest_path)
    train_ds = SurvivorDataset(df, "train", audio_cfg)
    val_ds = SurvivorDataset(df, "val", audio_cfg)
    test_ds = SurvivorDataset(df, "test", audio_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )
    return train_loader, val_loader, test_loader

