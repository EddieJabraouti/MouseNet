"""Test what TMT duration models do when all trial times are stretched.

This is an input-shift stress test, not validation on naturally longer
tasks. Synthetic copies stay within their source participant's training fold.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .model import DEMOGRAPHICS, ROOT, sha256
from .raw_trial_baselines import create_model
from .synthetic_raw_eval import real_profiles


OUT = ROOT / "exp1/runs/time_stretch_probe"
MENDELEY = ROOT / "data/mendeley/Behaviour Biometrics Dataset/raw_kmt_dataset"
TRIAL_AUDIT = ROOT / "exp1/runs/preprocessing_audit/trial_audit.csv"
SEEDS = tuple(range(42, 52))
SCALES = (1.0, 1.25, 1.5, 2.0, 4.0)
REGIMES = ("original", "augmented", "relative")
MODELS = ("rf", "xgb")


def mendeley_event_spans() -> np.ndarray:
    """Event spans of form-filling sessions; read no user identity or fields."""
    spans = []
    for path in sorted(MENDELEY.glob("raw_kmt_user_*.json")):
        with path.open() as handle:
            data = json.load(handle)
        for section in ("true_data", "false_data"):
            for trial in data[section].values():
                events = trial.get("mouse_events", []) + trial.get("key_events", [])
                stamps = [float(event["Epoch"]) for event in events if event.get("Epoch") is not None]
                if len(stamps) >= 2:
                    spans.append(max(stamps) - min(stamps))
    return np.asarray(spans, dtype=float)


def relative_durations(x: np.ndarray) -> np.ndarray:
    """Remove a uniform time multiplier while keeping all 20 trial values."""
    typical = np.median(x, axis=1, keepdims=True)
    if np.any(typical <= 0):
        raise ValueError("Nonpositive trial duration")
    return x / typical


def fit_model(name: str, regime: str, seed: int, x: np.ndarray, y: np.ndarray):
    model = create_model(name)
    model.set_params(random_state=seed)
    if regime == "augmented":
        # Source label is copied only within its training fold. Each source
        # retains total sample weight one despite the five stretched copies.
        train_x = np.concatenate([x * factor for factor in SCALES])
        train_y = np.tile(y, len(SCALES))
        weights = np.full(len(train_y), 1 / len(SCALES))
        if name == "rf":
            model.set_params(min_samples_leaf=8 * len(SCALES))
        model.fit(train_x, train_y, sample_weight=weights)
    else:
        train_x = relative_durations(x) if regime == "relative" else x
        model.fit(train_x, y)
    return model


def run(out_dir: Path = OUT):
    people, y, _, raw = real_profiles()
    duration = raw[:, :20]
    if duration.shape != (74, 20) or not np.isfinite(duration).all():
        raise ValueError("Expected 74 complete ordered TMT duration profiles")
    spans = mendeley_event_spans()
    if len(spans) != 1760 or not np.isfinite(spans).all():
        raise ValueError("Unexpected Mendeley session count or duration")

    predictions = []
    repeat_metrics = []
    for seed in SEEDS:
        fold_metrics = defaultdict(list)
        for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(people, y), 1):
            for model_name in MODELS:
                for regime in REGIMES:
                    model = fit_model(model_name, regime, seed, duration[train], y[train])
                    original_scores = None
                    for factor in SCALES:
                        x_test = duration[test] * factor
                        if regime == "relative":
                            x_test = relative_durations(x_test)
                        scores = model.predict_proba(x_test)[:, 1]
                        if factor == 1.0:
                            original_scores = scores
                        auroc = float(roc_auc_score(y[test], scores))
                        fold_metrics[(model_name, regime, factor)].append(auroc)
                        for position, i in enumerate(test):
                            predictions.append((seed, fold, people[i], int(y[i]), model_name, regime, factor,
                                                float(scores[position]), float(scores[position] - original_scores[position])))
        repeat_metrics.append({
            "seed": seed,
            "mean_fold_auroc": {
                f"{model}_{regime}_{factor:g}": float(np.mean(fold_metrics[(model, regime, factor)]))
                for model in MODELS for regime in REGIMES for factor in SCALES
            },
        })
        print(f"Completed time-stretch seed {seed}", flush=True)

    summary = {}
    for model in MODELS:
        for regime in REGIMES:
            for factor in SCALES:
                key = f"{model}_{regime}_{factor:g}"
                subset = [row for row in predictions if row[4] == model and row[5] == regime and row[6] == factor]
                repeat_means = [repeat["mean_fold_auroc"][key] for repeat in repeat_metrics]
                summary[key] = {
                    "mean_fold_auroc": float(np.mean(repeat_means)),
                    "min_repeat_mean": float(np.min(repeat_means)),
                    "max_repeat_mean": float(np.max(repeat_means)),
                    "mean_prediction": float(np.mean([row[7] for row in subset])),
                    "mean_absolute_prediction_change_from_original": float(np.mean([abs(row[8]) for row in subset])),
                    "fraction_predictions_crossing_0_5": float(np.mean([
                        (row[7] >= .5) != (row[7] - row[8] >= .5) for row in subset
                    ])),
                }

    report = {
        "scope": "Uniformly scale all 20 TMT trial durations for each held-out participant; original labels are retained only as a hypothetical robustness probe. This does not test a different task or provide real long-task cognitive outcomes.",
        "protocol": "Ten stratified participant-held-out five-fold shuffles, seeds 42-51. RF/XGB recipes fixed from raw-trial baseline. Original and relative models train on real TMT profiles only. Augmented model trains on each training participant's real profile and four scaled copies, equal total weight per source, with RF leaf count scaled by five. Every held-out person is real.",
        "scales": SCALES,
        "relative_representation": "Each of the 20 durations divided by the same participant's median duration. This is exactly invariant to uniform time scaling but removes absolute speed.",
        "n_tmt_people": len(people),
        "n_tmt_trials": duration.size,
        "tmt_trial_seconds_quantiles_0_25_50_75_100": [float(v) for v in np.quantile(duration, [0, .25, .5, .75, 1])],
        "fraction_tmt_trials_at_least_24_8_seconds": float(np.mean(duration >= 24.8)),
        "mendeley_note": "Mendeley event spans cover form-filling sessions with both mouse and keyboard events. They are a different task; long spans can include idle gaps and carry no comparable cognitive outcome. Used here only to show duration shift.",
        "n_mendeley_sessions": len(spans),
        "mendeley_event_span_seconds_quantiles_0_25_50_75_90_100": [float(v) for v in np.quantile(spans, [0, .25, .5, .75, .9, 1])],
        "fraction_mendeley_spans_over_25_seconds": float(np.mean(spans > 25)),
        "fraction_mendeley_spans_over_60_seconds": float(np.mean(spans > 60)),
        "trial_audit_sha256": sha256(TRIAL_AUDIT),
        "demographics_sha256": sha256(DEMOGRAPHICS),
        "summary": summary,
        "repeats": repeat_metrics,
        "limitation": "AUC on stretched copies of held-out TMT participants measures response to a synthetic input transformation, not transfer to real longer work. Uniformly stretching duration may itself change the disease-relevant signal, so label preservation cannot be assumed. Repeated splits reuse the same 74 people.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "outer_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("seed", "fold", "subject_id", "group", "model", "regime", "scale", "score", "change_from_original"))
        writer.writerows(predictions)
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k.endswith(("_1", "_2", "_4"))}, indent=2))


if __name__ == "__main__":
    run()
