import os
from typing import Dict

import torch
import torchaudio

from pipeline.config import AudioConfig
from pipeline.model import SmallAudioCNN


@torch.no_grad()
def infer_one(model_path: str, audio_path: str, audio_cfg: AudioConfig | None = None) -> Dict[str, float | str]:
    audio_cfg = audio_cfg or AudioConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt.get("audio_cfg")
    if cfg:
        audio_cfg = AudioConfig(**cfg)

    model = SmallAudioCNN(num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != audio_cfg.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, audio_cfg.sample_rate)

    target_len = audio_cfg.num_samples
    cur_len = wav.shape[-1]
    if cur_len < target_len:
        wav = torch.nn.functional.pad(wav, (0, target_len - cur_len))
    elif cur_len > target_len:
        start = (cur_len - target_len) // 2
        wav = wav[:, start : start + target_len]

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg.sample_rate,
        n_fft=audio_cfg.n_fft,
        hop_length=audio_cfg.hop_length,
        n_mels=audio_cfg.n_mels,
        f_min=audio_cfg.f_min,
        f_max=audio_cfg.f_max,
        power=2.0,
    )(wav)
    mel = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    x = mel.unsqueeze(0).to(device)

    probs = torch.softmax(model(x), dim=1).squeeze(0).cpu()
    pred = int(torch.argmax(probs).item())
    labels = ["non_distress", "distress"]
    return {
        "file": os.path.basename(audio_path),
        "prediction": labels[pred],
        "distress_prob": float(probs[1].item()),
        "non_distress_prob": float(probs[0].item()),
    }

