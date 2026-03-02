import argparse
import os
from dataclasses import asdict
from typing import Any, Dict, Union

import torch
import torchaudio

from pipeline.config import AudioConfig
from pipeline.model import SmallAudioCNN


class SoftmaxWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.base_model(x), dim=1)


def _deployment_target(coremltools_module, ios_version: int):
    target_map = {
        13: coremltools_module.target.iOS13,
        14: coremltools_module.target.iOS14,
        15: coremltools_module.target.iOS15,
        16: coremltools_module.target.iOS16,
        17: coremltools_module.target.iOS17,
        18: coremltools_module.target.iOS18,
    }
    if ios_version not in target_map:
        supported = ", ".join(str(k) for k in sorted(target_map))
        raise ValueError(f"Unsupported iOS target '{ios_version}'. Supported: {supported}")
    return target_map[ios_version]


def _infer_mel_frames(audio_cfg: AudioConfig) -> int:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg.sample_rate,
        n_fft=audio_cfg.n_fft,
        hop_length=audio_cfg.hop_length,
        n_mels=audio_cfg.n_mels,
        f_min=audio_cfg.f_min,
        f_max=audio_cfg.f_max,
        power=2.0,
    )
    dummy_wav = torch.zeros(1, audio_cfg.num_samples, dtype=torch.float32)
    return int(mel(dummy_wav).shape[-1])


def export_coreml(
    checkpoint_path: str,
    output_path: str,
    ios_target: int = 15,
    use_mlprogram: bool = True,
) -> Dict[str, Union[str, int, Dict[str, Any], list]]:
    try:
        import coremltools as ct
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'coremltools'. Install it first:\n"
            "python -m pip install coremltools"
        ) from exc

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("audio_cfg")
    audio_cfg = AudioConfig(**cfg) if cfg else AudioConfig()

    model = SmallAudioCNN(num_classes=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapped = SoftmaxWrapper(model).eval()
    mel_frames = _infer_mel_frames(audio_cfg)
    dummy = torch.randn(1, 1, audio_cfg.n_mels, mel_frames, dtype=torch.float32)
    traced = torch.jit.trace(wrapped, dummy)

    convert_to = "mlprogram" if use_mlprogram else "neuralnetwork"
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        convert_to=convert_to,
        minimum_deployment_target=_deployment_target(ct, ios_target),
        inputs=[
            ct.TensorType(
                name="mel_spectrogram",
                shape=dummy.shape,
                dtype=dummy.numpy().dtype,
            )
        ],
        outputs=[ct.TensorType(name="class_probs")],
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mlmodel.save(output_path)

    return {
        "checkpoint": os.path.abspath(checkpoint_path),
        "output": os.path.abspath(output_path),
        "format": convert_to,
        "ios_target": ios_target,
        "input_name": "mel_spectrogram",
        "input_shape": list(dummy.shape),
        "output_name": "class_probs",
        "audio_cfg": asdict(audio_cfg),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Export trained PyTorch audio model to CoreML.")
    p.add_argument("--checkpoint", default="artifacts/model/best_model.pth")
    p.add_argument("--output", default="artifacts/model/distress_classifier.mlpackage")
    p.add_argument("--ios_target", type=int, default=15)
    p.add_argument(
        "--neuralnetwork",
        action="store_true",
        help="Export legacy .mlmodel-style neuralnetwork instead of mlprogram.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    info = export_coreml(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        ios_target=args.ios_target,
        use_mlprogram=not args.neuralnetwork,
    )
    print(info)


if __name__ == "__main__":
    main()
