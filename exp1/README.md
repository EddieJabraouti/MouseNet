# MouseNet: experiment design

Working notes; update as the design develops. Design stage only: no training or
model results yet.

## Objective

Compare ways to classify mouse-movement batches as MCI-associated or
control-associated, then compare subsequent batches with a person's initial
baseline. Retain continuous classifier scores and movement representations so
we can measure both score changes and embedding differences.

## Data

Paths below are relative to the repository root.

- **TMT raw:** `data/nm9xy-osfstorage-raw_data-archive/`.
  Computerized Trail Making Test recordings for reconstructing ordered trials
  and time-series batches.
- **TMT processed:** `data/nm9xy-osfstorage-processed_data-archive/`.
  Contains `df_digital_tmt_with_target.csv`, `demographic_df.csv`, and
  `non_digital_df.csv`. Use these to establish clinical targets and participant
  joins; demographics and non-digital test scores are not inputs to the proposed
  mouse-only classifiers.
- **Mendeley:** `data/mendeley/Behaviour Biometrics Dataset/`.
  Includes `raw_kmt_dataset/` and `feature_kmt_dataset/`. Use mouse events from
  this authentication dataset for representation learning; it has no cognitive
  labels. The local directory is spelled `mendeley`.

The TMT study reports 74 analysed participants (41 MCI, 33 controls), approximately
60 Hz cursor coordinates and timestamps, and 20 experimental trials alternating
A/B after two practice trials. Each trial lasts at most 25 seconds and may finish
early. Diagnosis is a participant-level label, not a changing label per movement.
These are published specifications; local schemas, usable counts, timing, and
label joins still need auditing.

## Experiments

1. **Pretrained temporal encoder + frozen embeddings** (`pte_classifier/`).
   Pretrain a temporal encoder on Mendeley mouse sequences. Freeze it, encode
   TMT batches into fixed-length vectors, and train a supervised MCI/control
   classifier using only those embeddings. Pretraining objective, encoder
   architecture, embedding size, and classifier remain open.
2. **Direct supervised TMT model** (`supervised_class/`).
   Train a temporal model from scratch on labelled TMT sequences. Compare a
   real-data-only arm with an augmented-training arm. Prefer the same temporal
   backbone as experiment 1 to make architecture differences less confounding.
   Mendeley may inform a generator or perturbation model, but cannot simply be
   added as MCI/control examples. Label preservation must be tested; augmentation
   adds training examples, not independent clinical participants.
3. **Optional heatmap CNN** (`2D_CNN/`).
   Render each batch as a spatial representation and train a CNN. Consider dwell
   time and temporal progression channels because a plain occupancy heatmap
   discards movement order. Keep this as a later comparison.

## Baseline monitoring

- Establish a reference from the first `n` chronological batches, storing their
  scores, score variability, and embeddings where available.
- Compare later groups of batches against that fixed reference:
  `score_delta = mean(follow_up_scores) - mean(baseline_scores)`.
- Also compare embedding distributions; a behavioural change may occur without
  a large classifier-score change. Enrollment does not imply healthy status.
- Keep the model and preprocessing fixed. Use comparable task composition:
  complete A/B pairs are an initial candidate, while fixed-duration windows
  remain an alternative. Actual trial durations vary, so enrollment time is
  the sum of batch durations unless fixed-length windows are used.

## Shared evaluation and open decisions

- Share participant splits, batch definitions, and metrics across experiments.
  Keep all trials, windows, and synthetic descendants of a person in one split.
- Fit preprocessing and augmentation on training data only; evaluate on real
  held-out participants. Personal enrollment uses only their earlier batches.
- Compare participant-level AUROC and balanced accuracy, plus within-person
  score stability and deviation behaviour. Choose tuning and aggregation rules
  before inspecting held-out results.
- Decide window length versus whole trials/A-B pairs, baseline batch count,
  pretraining loss, classifier, augmentation methods, and deviation thresholds.
- This cross-sectional dataset permits within-session replay, not validation
  of longitudinal cognitive decline. Practice, task layout, fatigue, and device
  context can affect deviations. Transfer to everyday use remains a hypothesis.
- Check TMT reuse terms: its OSF landing page currently lists no license.

## Sources

- [TMT data](https://osf.io/nm9xy/)
- [TMT study](https://www.nature.com/articles/s41598-026-62955-9)
- [NeuroLIAA analysis code](https://github.com/NeuroLIAA/tmt-analysis)
- [Mendeley behavioural biometrics](https://data.mendeley.com/datasets/fnf8b85kr6/1)
