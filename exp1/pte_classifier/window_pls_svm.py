"""Window-level frozen Mouse2Vec -> supervised PLS -> RBF SVM.

No window or trial embeddings are pooled before PLS or classification. All
model-selection splits are by participant. Participant scores are the median
of their *post-classifier* window decision scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .model import DEMOGRAPHICS, ROOT, RUN_DIR, SEED, demographic_features, demographic_targets, sha256
from .window_model import WindowPLSSVM


OUT_DIR = ROOT / "exp1/runs/window_pls_rbf"
COMPONENTS = (2, 4, 8)
C_VALUES = (0.1, 1.0)


def load_windows(path: Path):
    with np.load(path, allow_pickle=False) as data:
        z = data["embeddings"].astype(np.float64)
        window_ids = data["subject_id"]
        trial_type = data["trial_type"]
    people = np.unique(window_ids)
    targets = demographic_targets()
    if z.shape != (26655, 128) or len(people) != 74 or set(people) != set(targets):
        raise ValueError("Window embeddings fail cohort audit")
    if not np.isfinite(z).all() or set(trial_type) != {"A", "B"}:
        raise ValueError("Invalid window embeddings or TMT part labels")
    y_people = np.asarray([targets[sid] for sid in people], dtype=int)
    if Counter(y_people) != Counter({0: 33, 1: 41}):
        raise ValueError("Unexpected participant class counts")
    y_windows = np.asarray([targets[sid] for sid in window_ids], dtype=int)
    demographics = demographic_features(people, y_people)
    demo_by_id = {sid: demographics[i] for i, sid in enumerate(people)}
    demo_windows = np.asarray([demo_by_id[sid] for sid in window_ids])
    return z, window_ids, y_windows, people, y_people, demo_windows, demographics


def participant_scores(model: WindowPLSSVM, z, demographics, window_ids, people):
    scores = model.decision_function(z, demographics)
    return np.asarray([np.median(scores[window_ids == sid]) for sid in people])


def choose_parameters(z, window_ids, y_windows, people, y_people, demo_windows):
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = list(inner.split(people, y_people))
    results = []
    for n_components in COMPONENTS:
        for C in C_VALUES:
            aucs = []
            for train, valid in folds:
                train_mask = np.isin(window_ids, people[train])
                valid_mask = np.isin(window_ids, people[valid])
                model = WindowPLSSVM(n_components, C).fit(
                    z[train_mask], demo_windows[train_mask], y_windows[train_mask], window_ids[train_mask]
                )
                scores = participant_scores(
                    model, z[valid_mask], demo_windows[valid_mask], window_ids[valid_mask], people[valid]
                )
                aucs.append(float(roc_auc_score(y_people[valid], scores)))
            results.append({"n_components": n_components, "C": C, "inner_mean_auroc": float(np.mean(aucs))})
    # Explicit tie-breaking prefers fewer PLS components, then smaller C.
    return sorted(results, key=lambda row: (-row["inner_mean_auroc"], row["n_components"], row["C"]))[0], results


def evaluate(windows_path: Path, out_dir: Path):
    z, window_ids, y_windows, people, y_people, demo_windows, _ = load_windows(windows_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_scores = np.full(len(people), np.nan)
    folds_report = []
    for fold_number, (train, test) in enumerate(outer.split(people, y_people), 1):
        train_mask = np.isin(window_ids, people[train])
        test_mask = np.isin(window_ids, people[test])
        if np.any(train_mask & test_mask):
            raise AssertionError("Participant overlap in outer fold")
        selected, _ = choose_parameters(
            z[train_mask], window_ids[train_mask], y_windows[train_mask], people[train], y_people[train], demo_windows[train_mask]
        )
        model = WindowPLSSVM(selected["n_components"], selected["C"]).fit(
            z[train_mask], demo_windows[train_mask], y_windows[train_mask], window_ids[train_mask]
        )
        scores = participant_scores(model, z[test_mask], demo_windows[test_mask], window_ids[test_mask], people[test])
        oof_scores[test] = scores
        folds_report.append(
            {
                "fold": fold_number,
                "n_test_participants": len(test),
                "n_components": selected["n_components"],
                "C": selected["C"],
                "inner_mean_auroc": selected["inner_mean_auroc"],
                "test_auroc": float(roc_auc_score(y_people[test], scores)),
                "test_balanced_accuracy": float(balanced_accuracy_score(y_people[test], scores >= 0)),
            }
        )
        print(f"Evaluated participant fold {fold_number}/5", flush=True)
    if not np.isfinite(oof_scores).all():
        raise ValueError("Missing participant prediction")
    with (out_dir / "outer_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject_id", "group", "median_window_decision_score", "predicted_group"))
        writer.writerows(zip(people, y_people, oof_scores, (oof_scores >= 0).astype(int)))

    selected, selection_results = choose_parameters(z, window_ids, y_windows, people, y_people, demo_windows)
    final = WindowPLSSVM(selected["n_components"], selected["C"]).fit(z, demo_windows, y_windows, window_ids)
    joblib.dump(final, out_dir / "final_classifier.joblib")
    report = {
        "pipeline": "each frozen 128D Mouse2Vec window embedding -> fold-fitted PLS -> RBF SVM; demographics join after PLS",
        "embedding_aggregation": "none before PLS or SVM",
        "participant_score": "median of post-classifier window decision scores; zero is the decision threshold",
        "validation": "outer stratified 5-fold by participant; inner stratified 3-fold by participant; inner selection uses mean validation-fold participant AUROC",
        "model_selection": {"components": COMPONENTS, "C": C_VALUES, "gamma": "scale"},
        "n_participants": len(people),
        "n_windows": len(z),
        "class_counts": dict(Counter(int(v) for v in y_people)),
        "mean_fold_auroc": float(np.mean([row["test_auroc"] for row in folds_report])),
        "mean_fold_balanced_accuracy": float(np.mean([row["test_balanced_accuracy"] for row in folds_report])),
        "pooled_oof_auroc_descriptive": float(roc_auc_score(y_people, oof_scores)),
        "pooled_oof_balanced_accuracy": float(balanced_accuracy_score(y_people, oof_scores >= 0)),
        "folds": folds_report,
        "final_parameters": selected,
        "final_inner_search": selection_results,
        "mouse_features_sha256": sha256(windows_path),
        "demographics_sha256": sha256(DEMOGRAPHICS),
        "limitations": "PLS treats windows as separate labeled observations and does not model their order across windows. Training weights keep participant influence equal in the SVM, but PLS itself is fitted on every training window without participant weights. No independent test cohort.",
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=RUN_DIR / "window_embeddings.npz")
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    evaluate(args.windows, args.output)
