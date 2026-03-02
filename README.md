# audio_event_detection

PyTorch pipeline to train a **binary audio classifier** for **distress vs non-distress** events from a folder-based WAV dataset.

## Requirements

- Python **3.10+** (repo code uses modern type syntax; tested here with Python 3.11)
- `pip` dependencies: see `requirements.txt`

Install:

```bash
python3 -m pip install -r requirements.txt
```

## Dataset layout

By default the CLI expects a dataset root like `data/DisasterDataset/` with:

```text
data/DisasterDataset/
  reference/
    <event_name>/*.wav
  generated/                # optional (used for train split only)
    <event_name>/*.wav
```

Where each `<event_name>` is a folder name (for example: `screaming`, `cough`, `wind`, ...).

### Distress labels (positive class)

During manifest creation, each audio clip is labeled as:

- `label_id = 1` (`label_name = distress`) if its folder name matches one of the **positive events**
- `label_id = 0` (`label_name = non_distress`) otherwise

Default positive events are defined in `pipeline/cli.py` / `pipeline/data.py`:

- `asking help`
- `screaming`
- `crying sobbing wail`
- `whispering`
- `cough`

If your dataset uses different folder names, pass them via `--positive_events`.

## Training

Training is a two-step process:

1) Build a `manifest.csv` (train/val/test split + labels)
2) Train a small CNN on mel-spectrograms

All commands below are meant to be run from the repo root.

### 1) Prepare the manifest

```bash
python3 -m pipeline.cli prepare \
  --dataset_root data/DisasterDataset \
  --out_dir artifacts/data \
  --generated_ratio 1.0 \
  --val_split 0.15 \
  --test_split 0.10 \
  --seed 42
```

Outputs:

- `artifacts/data/manifest.csv`
- `artifacts/data/manifest_summary.json`

Notes:

- The manifest stores **absolute** WAV paths. If you move the dataset, re-run `prepare`.
- Validation/test splits are made from `reference/` only; `generated/` (if present) is sampled into the **train** split only.
- Control how much synthetic/generated data is added with `--generated_ratio`:
  - `0` = ignore `generated/`
  - `1.0` = cap generated samples per class to ~the number of reference train samples per class

### 2) Train the model

```bash
python3 -m pipeline.cli train \
  --manifest artifacts/data/manifest.csv \
  --out_dir artifacts/model \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-3 \
  --distress_weight 2.0 \
  --num_workers 2
```

Outputs:

- `artifacts/model/best_model.pth` (checkpoint with `model_state_dict` and `audio_cfg`)
- `artifacts/model/train_history.json`
- `artifacts/model/test_metrics.json`

The trainer automatically uses CUDA if available, otherwise CPU (`pipeline/train.py`).

## Inference (optional)

Run inference on a single WAV file:

```bash
python3 -m pipeline.cli infer \
  --model artifacts/model/best_model.pth \
  --audio path/to/audio.wav
```

## What the model sees (audio preprocessing)

Preprocessing is defined in `pipeline/config.py`, `pipeline/data.py`, and `pipeline/infer.py`:

- Resample to `sample_rate = 16000`
- Convert to mono (average channels) if needed
- Center crop / pad to `clip_seconds = 3.0`
- Convert to a mel-spectrogram (`n_mels = 64`, `n_fft = 1024`, `hop_length = 256`)
- Convert power to dB and normalize per-clip

## Using your own dataset

Option A (recommended): follow the folder structure in **Dataset layout** and run `prepare`.

Option B: create your own manifest CSV and point `train` at it. The training dataloader requires at least:

- `path`: absolute or relative path to a `.wav`
- `split`: one of `train` / `val` / `test`
- `label_id`: `0` or `1`

Extra columns like `event`, `source`, and `label_name` are optional (they’re included by `prepare`).

## Troubleshooting

- If `prepare` fails with stratification errors, it usually means one split/class is missing. Check that your `--positive_events` actually match folder names and that you have enough clips per class.
- If you hit audio loading issues, verify files are readable WAVs and that `soundfile` installed correctly (it’s included in `requirements.txt`).

