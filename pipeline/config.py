from dataclasses import dataclass


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    clip_seconds: float = 3.0
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    f_min: float = 20.0
    f_max: float = 8000.0

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.clip_seconds)


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.15
    test_split: float = 0.10
    num_workers: int = 2
    seed: int = 42
    distress_weight: float = 2.0

