import json
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from pipeline.config import AudioConfig, TrainConfig
from pipeline.data import create_dataloaders
from pipeline.model import SmallAudioCNN


def _metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int, int, int]:
    preds = logits.argmax(dim=1)
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    return tp, fp, tn, fn


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = 0
    tp = fp = tn = fn = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item())
        batches += 1
        btp, bfp, btn, bfn = _metrics_from_logits(logits, y)
        tp += btp
        fp += bfp
        tn += btn
        fn += bfn

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "loss": total_loss / max(batches, 1),
        "accuracy": acc,
        "distress_precision": precision,
        "distress_recall": recall,
        "false_negative_rate": fn / max(fn + tp, 1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def train_model(
    manifest_path: str,
    output_dir: str,
    audio_cfg: AudioConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> Dict[str, str]:
    audio_cfg = audio_cfg or AudioConfig()
    train_cfg = train_cfg or TrainConfig()
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = create_dataloaders(manifest_path, audio_cfg, train_cfg)

    model = SmallAudioCNN(num_classes=2).to(device)
    class_weights = torch.tensor([1.0, train_cfg.distress_weight], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    best_val_loss = float("inf")
    best_path = os.path.join(output_dir, "best_model.pth")
    history = []

    for epoch in range(train_cfg.epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_cfg.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(batches, 1)
        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val": val_metrics}
        history.append(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "audio_cfg": vars(audio_cfg),
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    history_path = os.path.join(output_dir, "train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    return {
        "best_model": best_path,
        "history": history_path,
        "test_metrics": metrics_path,
    }

