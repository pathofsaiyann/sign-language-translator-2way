# GISLR participant-aware extraction

Run from `sign_language_backend`:

```sh
../.venv/bin/python -m pip install -r requirements-gislr.txt
../.venv/bin/python scripts/extract_gislr.py --download --prompt-token
# Or use the unzipped Kaggle competition directory:
../.venv/bin/python scripts/extract_gislr.py --dataset-dir /path/to/asl-signs
```

The directory must contain `train.csv`, `sign_to_prediction_index_map.json`,
and the parquet files referenced by `train.csv`. Both metadata files are read;
source indices are recorded as provenance. Output indices 0–29 follow the exact
requested vocabulary order in `WORDS`. Missing signs are reported without aliases
or substitutes. Coverage is determined from source metadata, never assumed.

A seeded global participant split reserves 20% of target-data participants for
validation (`--val-fraction`, default .2; `--seed`, default 42). A participant is
assigned to only one split across every class. Small signer groups can prevent
class coverage in one split; these shortfalls are reported, never repaired by
moving individual sequences across splits. Per-class caps and quotas have been
removed; every target CSV row is considered.

By default all matching sequences are extracted. Files are saved to
`data/processed/{train,val}/{class}/{sequence_id}.npy`. Features are XYZ for pose
indices 0–17, left hand 0–20, right hand 0–20. These 18 pose points include face
points and exclude hips/legs. Invalid or missing landmarks are zero-filled.
Frames are sorted by frame ID; sequences above 64 frames are uniformly
subsampled with both endpoints retained; short sequences receive trailing zeros.
Every saved array is finite float32 `(64,180)`. Inference runs on the WebSocket
server with matching shoulder normalization; the webcam only streams JPEGs.

`metadata.json` schema v2 includes:

- `class_to_index`: all 30 requested labels and their fixed output indices.
- `samples`: project-relative NPY paths mapped to output class indices.
- `sample_info`: source path/index, sequence ID, participant, class and split.
- `participants`: disjoint assigned signer lists for train and validation.
- `stats`: per-class counts, candidate counts, split totals, errors and settings.

`data/gislr_extraction_report.json` mirrors stats; `data/gislr_labels.json` stores
labels. This schema replaces the earlier flat metadata mapping; training code
must read `metadata['samples']` and honor the recorded split. Reruns validate and reuse existing indexed tensors with matching CSV provenance
and participant assignment, then add missing samples. Existing data is retained
if a download fails, and metadata records the partial expansion. Use metadata
rather than directory scans. Extraction appends metrics to context/progress.

Authentication supports a hidden process-only prompt, `KAGGLE_API_TOKEN`, legacy
`KAGGLE_USERNAME`/`KAGGLE_KEY`, and `~/.kaggle/kaggle.json` (also
`KAGGLE_CONFIG_DIR`). Tokens are not saved. Download errors print sanitized status
and CLI fallback instructions, then exit 1. Accept the `asl-signs` competition
rules using the credential's account. Exit 2 indicates unreadable/invalid samples.
Extraction completeness and vocabulary coverage are reported separately: 19
source classes can be fully extracted even though 11 original labels are absent. Failure before source metadata
is loaded leaves reset metadata untouched; no split can be claimed in that case.

Downloads use four bounded workers by default (`--workers` 1–16). HTTP 429
and temporary server errors receive up to five retries with 30-second waits.
Cached source files are reused on reruns; selection and splits remain deterministic.

## Regularized training and inference contract

Run `../.venv/bin/python scripts/train_gislr.py --epochs 70` from the backend.
The original participant assignment is retained (17 train / 4 validation signers,
19 present labels); sample counts expand with the dataset.
Disk NPY tensors remain raw XYZ `(64,180)`; preprocessing is in memory.

Shared `scripts/_shared/preprocessing.py` centers every active joint on the pose
shoulder midpoint (indices 11/12) and divides all XYZ by their 3D distance.
Missing/nonfinite joint triples stay zero. Frames without both valid shoulders,
or with shoulder distance <.001, are entirely masked; no temporal anchor imputation
is used. This policy avoids scale explosions from invalid or collapsed anchors.

Training Dataset access clones normalized tensors and applies sequence-level
scale .9–1.1, per-axis translation -.05–.05 and Gaussian noise with std .015 in
normalized coordinates. Only active joints are transformed. Seven to nine random
positions out of 64 are masked (padding positions may be selected); at least one
active frame survives. Original sequence duration is retained for packing.
Validation only uses deterministic normalization; no random augmentation.

BiLSTM dropout is .4, CrossEntropyLoss label_smoothing=.1, AdamW lr=.001 with
weight_decay=.001, ReduceLROnPlateau max/factor=.5/patience=5. Default is 70 epochs.
History contains augmentation, preprocessing and optimizer settings plus all epoch
metrics. Training accuracies are measured with augmentation and dropout enabled.
Previous exported assets/history are backed up in timestamped data/training_runs/.

Deploy the best weights together with labels_19class.json and
model_gislr_19class.config.json in models/. The server requires the config and
applies the exact same normalization before model inference. Its motion gate still
uses raw coordinates, preserving the existing .015 threshold semantics. Restart
any running server after training; it holds the old model/config until restarted.
The webcam remains a JPEG WebSocket client and requires no preprocessing changes.

Earlier capped-data 70-epoch run: best validation 37.50% at epoch 33 (previous best 34.21%).
Final augmented training 98.68%, validation 32.24%; overfitting persists despite
the modest validation improvement. This is not evidence of improved live-camera
accuracy; real-signer validation and broader training data are still needed.

## Full target-source extraction

There are 7,327 indexed sequences in the 19 available target classes: 5,967 train
and 1,360 val under seed 42 / .2 participant split, before invalid-sample checks.
The old 32/8 sample quota and --samples-per-class option are removed.

Use the supported extractor against cached or downloaded official parquet files:

```sh
../.venv/bin/python scripts/extract_gislr.py --download --prompt-token
../.venv/bin/python scripts/train_gislr.py --epochs 50
```

The experimental signed-archive range helper was removed during cleanup; core
extraction uses bounded per-file downloads and cached files. No existing cached
parquet files or tensors were removed.

## Completed full-dataset 50-epoch run (2026-11-09)

All 7,327 target parquet sequences converted successfully: 5,967 train, 1,360
validation, 17/4 disjoint participants, 19 available classes. No source failures,
invalid normalization samples or unindexed NPY files were found. Extraction and
vocabulary completeness are separate; 11 original vocabulary labels are absent
from the source, but extraction of the 19 present classes is complete.

The 50-epoch regularized BiLSTM run saved its best checkpoint at epoch 45:
validation accuracy 71.1029% (967/1360), smoothed cross-entropy loss 1.526864.
Final augmented training accuracy was 98.3912% (loss .662513); final validation
accuracy 67.1324% (loss 1.598593). A generalization gap remains. These validation
scores use the expanded split and are not directly comparable to capped-data
scores measured on only 152 validation samples. Live-camera accuracy is untested.

All exports were updated and the best checkpoint reloaded successfully. Prior
model/history assets are in data/training_runs/20261109T134135506757Z/; pre-expansion
metadata is in data/extraction_runs/20261109T125500Z/. Restart the server to load
the new checkpoint/config. Verification: 6 extraction, 2 archive, 4 training and
14 server tests passed, plus full-dataset provenance/shape/normalization audits.
