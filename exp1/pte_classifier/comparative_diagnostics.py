"""Post-hoc age-matched ranking diagnostics for completed held-out runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .model import ROOT, demographic_features, demographic_targets
from .temporal_experiment import age_matched_pair_ranking


OUT = ROOT / "exp1/runs/comparative_diagnostics"
RUNS = {
    "raw_duration20_rf": (ROOT / "exp1/runs/raw_trial_baselines/outer_predictions.csv", "model", "duration20_rf"),
    "raw_duration20_xgb": (ROOT / "exp1/runs/raw_trial_baselines/outer_predictions.csv", "model", "duration20_xgb"),
    "raw60_rf": (ROOT / "exp1/runs/raw_trial_baselines/outer_predictions.csv", "model", "raw60_rf"),
    "raw60_xgb": (ROOT / "exp1/runs/raw_trial_baselines/outer_predictions.csv", "model", "raw60_xgb"),
    "processed_combined_logistic": (ROOT / "exp1/runs/processed_feature_baselines/outer_predictions.csv", "model", "combined_logistic"),
    "stable_mouse_xgb_pls2_combined": (ROOT / "exp1/runs/regularized_trees/stable_hold_window/outer_predictions.csv", "model", "xgb_pls2_combined"),
    "stable_mouse_rf_direct_mouse": (ROOT / "exp1/runs/regularized_trees/stable_hold_window_direct/outer_predictions.csv", "model", "rf_direct_mouse"),
    "demographics_rf": (ROOT / "exp1/runs/processed_feature_baselines/outer_predictions.csv", "model", "demographics_rf"),
}


def run(out_dir: Path = OUT):
    targets = demographic_targets()
    people = np.array(sorted(targets))
    y = np.array([targets[sid] for sid in people])
    ages = demographic_features(people, y)[:, 1]
    results = {}
    for name, (path, model_field, selected) in RUNS.items():
        with path.open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row[model_field] == selected]
        by_id = {row["subject_id"]: row for row in rows}
        if len(rows) != 74 or set(by_id) != set(people):
            raise ValueError(f"Incomplete held-out predictions for {name}")
        folds = np.array([int(by_id[sid]["fold"]) for sid in people])
        score = np.array([float(by_id[sid]["score"]) for sid in people])
        if any(int(by_id[sid]["group"]) != targets[sid] for sid in people):
            raise ValueError("Label mismatch")
        results[name] = age_matched_pair_ranking(y, ages, score, folds, tolerance_years=5)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "Within each held-out fold, compare MCI/control prediction scores only for pairs whose ages differ by at most five years; no model fitting or tuning",
        "results": results,
        "caution": "Pairs share participants and are correlated. These are post-hoc descriptive rankings on a reused cohort, not an age-matched external validation or independent sample size.",
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
