"""Controlled uniform time-scaling test when synthetic task identity is known.

Each synthetic task is exactly the same TMT profile multiplied by a fixed
factor. This tests whether task-conditioned models can preserve rankings under
that assumption; it is not validation on a genuinely different task.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .model import ROOT
from .raw_trial_baselines import create_model
from .synthetic_raw_eval import real_profiles


OUT = ROOT / "exp1/runs/task_conditioned_stretch"
SEEDS = tuple(range(42, 52))
SCALES = (1.0, 1.5, 2.0, 4.0)
MODELS = ("rf", "xgb")


def fit(name: str, seed: int, x: np.ndarray, y: np.ndarray, scale: float = 1.0):
    model = create_model(name)
    model.set_params(random_state=seed)
    return model.fit(x * scale, y)


def task_block(x: np.ndarray, task_index: int) -> np.ndarray:
    one_hot = np.zeros((len(x), len(SCALES)), dtype=float)
    one_hot[:, task_index] = 1.0
    return np.column_stack((x * SCALES[task_index], one_hot))


def fit_pooled(name: str, seed: int, x: np.ndarray, y: np.ndarray):
    model = create_model(name)
    model.set_params(random_state=seed)
    if name == "rf":
        model.set_params(min_samples_leaf=8 * len(SCALES))
    features = np.concatenate([task_block(x, index) for index in range(len(SCALES))])
    labels = np.tile(y, len(SCALES))
    weights = np.full(len(labels), 1 / len(SCALES))
    model.fit(features, labels, sample_weight=weights)
    return model


def run(out_dir: Path = OUT):
    people, labels, _, raw = real_profiles()
    time = raw[:, :20]
    if time.shape != (74, 20):
        raise ValueError("Expected 74 complete TMT time profiles")
    predictions = []
    repeats = []
    for seed in SEEDS:
        fold_metrics = defaultdict(list)
        for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(people, labels), 1):
            for name in MODELS:
                baseline = fit(name, seed, time[train], labels[train])
                pooled = fit_pooled(name, seed, time[train], labels[train])
                base_scores = baseline.predict_proba(time[test])[:, 1]
                for task_index, scale in enumerate(SCALES):
                    separate = fit(name, seed, time[train], labels[train], scale)
                    cases = {
                        "unconditioned": baseline.predict_proba(time[test] * scale)[:, 1],
                        "separate_task": separate.predict_proba(time[test] * scale)[:, 1],
                        "pooled_task_label": pooled.predict_proba(task_block(time[test], task_index))[:, 1],
                        "known_scale_normalized": baseline.predict_proba(time[test] * scale / scale)[:, 1],
                    }
                    for regime, scores in cases.items():
                        fold_metrics[(name, regime, scale)].append(float(roc_auc_score(labels[test], scores)))
                        for i, score, base_score in zip(test, scores, base_scores):
                            predictions.append((seed, fold, people[i], int(labels[i]), name, regime, scale,
                                                float(score), float(score - base_score)))
        repeats.append({
            "seed": seed,
            "mean_fold_auroc": {
                f"{name}_{regime}_{scale:g}": float(np.mean(fold_metrics[(name, regime, scale)]))
                for name in MODELS
                for regime in ("unconditioned", "separate_task", "pooled_task_label", "known_scale_normalized")
                for scale in SCALES
            },
        })
        print(f"Completed task-conditioned seed {seed}", flush=True)

    summary = {}
    for name in MODELS:
        for regime in ("unconditioned", "separate_task", "pooled_task_label", "known_scale_normalized"):
            for scale in SCALES:
                key = f"{name}_{regime}_{scale:g}"
                selected = [p for p in predictions if p[4] == name and p[5] == regime and p[6] == scale]
                aucs = [row["mean_fold_auroc"][key] for row in repeats]
                summary[key] = {
                    "mean_fold_auroc": float(np.mean(aucs)),
                    "min_repeat_mean": float(np.min(aucs)),
                    "max_repeat_mean": float(np.max(aucs)),
                    "mean_absolute_score_difference_from_original": float(np.mean([abs(p[8]) for p in selected])),
                }

    report = {
        "question": "Does a fixed scale applied equally to both clinical groups preserve discrimination when the synthetic task scale is known at training or inference?",
        "scales": SCALES,
        "protocol": "Ten participant-held-out five-fold shuffles, seeds 42-51. Every test person is a real TMT participant whose original 20-time profile is uniformly multiplied for this hypothetical probe. Separate-task model trains on the same scale. Pooled model uses one-hot synthetic task labels and gives all copies of a training person total weight one. Known-scale model divides by that task's exact factor before inference. No synthetic source enters a held-out training fold.",
        "n_people": len(people),
        "feature_count": 20,
        "summary": summary,
        "repeats": repeats,
        "limitation": "Synthetic task labels encode only the imposed multiplier; no task demands, completion mechanism, trajectories, or disease effects change. Success here establishes mathematical/algorithmic behavior under exact uniform scaling, not transfer to real longer tasks or a new single-task model.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "outer_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("seed", "fold", "subject_id", "group", "model", "regime", "scale", "score", "difference_from_original"))
        writer.writerows(predictions)
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key.endswith("_4")}, indent=2))


if __name__ == "__main__":
    run()
