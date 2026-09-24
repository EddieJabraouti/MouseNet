"""Repeat fixed participant folds to assess sensitivity of leading baselines."""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .model import ROOT, demographic_features
from .processed_baselines import DATA, estimator
from .raw_trial_baselines import create_model
from .synthetic_raw_eval import real_profiles, synthetic_profiles
from .synthetic_tmt import OUT as SYNTH_DIR


OUT = ROOT / "exp1/runs/stability_checks"
SEEDS = tuple(range(42, 52))


def run(out_dir: Path = OUT):
    warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning, module="sklearn.utils.extmath")
    people, y, demo, raw = real_profiles()
    processed = pd.read_csv(DATA, dtype={"subject_id": str})
    by_id = processed.set_index("subject_id").loc[people]
    if not np.array_equal(by_id.group.to_numpy(dtype=int), y):
        raise ValueError("Processed/raw label mismatch")
    digital = by_id.drop(columns="group").to_numpy(dtype=float)
    combined = np.column_stack((digital, demo))
    synth_ids, source, sy, _, sX = synthetic_profiles(SYNTH_DIR / "synthetic_raw_trajectories.npz")
    if len(synth_ids) != 148:
        raise ValueError("Synthetic profile count mismatch")
    labels_by_id = dict(zip(people, y))
    if any(sy[i] != labels_by_id[sid] for i, sid in enumerate(source)):
        raise ValueError("Synthetic source label mismatch")
    feature_sets = {
        "duration20_rf": raw[:, :20],
        "duration20_rf_synthetic": raw[:, :20],
        "duration20_xgb": raw[:, :20],
        "raw60_rf": raw,
        "raw60_xgb": raw,
        "processed_combined_logistic": combined,
        "demographics_rf": demo,
    }
    repeats = []
    for seed in SEEDS:
        fold_metrics = {key: [] for key in feature_sets}
        for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(people, y):
            for key, X in feature_sets.items():
                if key == "duration20_rf_synthetic":
                    allowed = np.isin(source, people[train])
                    if np.any(np.isin(source[allowed], people[test])):
                        raise AssertionError("Synthetic descendant leaked into training")
                    lineage = np.concatenate((people[train], source[allowed]))
                    counts = Counter(lineage)
                    class_counts = Counter(y[train])
                    weights = np.array([1 / (2 * class_counts[labels_by_id[sid]] * counts[sid]) for sid in lineage])
                    weights *= len(train) / weights.sum()
                    model = create_model("rf")
                    model.set_params(min_samples_leaf=int(np.ceil(8 * len(lineage) / len(train))), class_weight=None)
                    model.fit(np.concatenate((X[train], sX[allowed, :20])), np.concatenate((y[train], sy[allowed])), sample_weight=weights)
                else:
                    model = estimator("logistic") if key.endswith("logistic") else create_model("xgb" if key.endswith("xgb") else "rf")
                    model.fit(X[train], y[train])
                score = model.decision_function(X[test]) if key.endswith("logistic") else model.predict_proba(X[test])[:, 1]
                fold_metrics[key].append(float(roc_auc_score(y[test], score)))
        repeats.append({"split_seed": seed, "mean_fold_auroc": {key: float(np.mean(values)) for key, values in fold_metrics.items()}, "fold_aurocs": fold_metrics})
        print(f"Finished repeated split {seed}", flush=True)
    summary = {
        key: {
            "mean_across_repeated_splits": float(np.mean([r["mean_fold_auroc"][key] for r in repeats])),
            "min_repeat": float(min(r["mean_fold_auroc"][key] for r in repeats)),
            "max_repeat": float(max(r["mean_fold_auroc"][key] for r in repeats)),
            "seed42_primary_split": repeats[0]["mean_fold_auroc"][key],
        }
        for key in feature_sets
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "10 repeats of stratified 5-fold participant CV with unchanged fixed models; split seeds 42-51. Repeats reuse the same 74 people and do not create independent test cohorts.",
        "models": {"duration20_rf": "300 trees, max depth 3, min leaf 8", "duration20_rf_synthetic": "same RF with 2 training-only descendants per source and scaled leaf size", "duration20_xgb": "150 trees, depth 2, min child weight 2, gamma 0.5, lambda 10, alpha 1", "raw60_rf": "same RF", "raw60_xgb": "same XGB", "processed_combined_logistic": "L2 C=0.1", "demographics_rf": "same RF"},
        "summary": summary, "repeats": repeats,
        "selection_caution": "These are exploratory stability checks after inspecting the seed-42 split; the range is not a confidence interval for external performance.",
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    run()
