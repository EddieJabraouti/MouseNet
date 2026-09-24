"""Classify ordered raw TMT trial timing/path/click descriptors.

This preserves all 20 trial values per channel and never pools Mouse2Vec
embeddings. It is a diagnostic of information the frozen encoder never sees
directly, not a replacement for a general-use mouse representation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .audit_preprocessing import OUT as AUDIT_DIR
from .model import DEMOGRAPHICS, ROOT, SEED, demographic_features, demographic_targets, sha256


OUT = ROOT / "exp1/runs/raw_trial_baselines"


def create_model(name):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight="balanced", solver="liblinear"))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=8, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=4)
    if name == "xgb":
        return XGBClassifier(n_estimators=150, learning_rate=.03, max_depth=2, min_child_weight=2, gamma=.5, reg_lambda=10, reg_alpha=1, subsample=.8, colsample_bytree=.8, tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=4)
    raise ValueError(name)


def evaluate(out_dir: Path = OUT):
    path = AUDIT_DIR / "trial_audit.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    targets = demographic_targets()
    people = np.array(sorted(targets))
    y = np.array([targets[sid] for sid in people])
    demo = demographic_features(people, y)
    by_id = {sid: [] for sid in people}
    for row in rows:
        by_id[row["subject_id"]].append(row)
    X = []
    for sid in people:
        trials = sorted(by_id[sid], key=lambda row: int(row["trial_order"]))
        if len(trials) != 20 or [int(row["trial_order"]) for row in trials] != list(range(20)):
            raise ValueError("Raw trial order mismatch")
        X.append([[float(row[field]) for row in trials] for field in ("duration_s", "raw_path", "clicks")])
    X = np.asarray(X, dtype=float)
    durations = X[:, 0, :]
    raw60 = X.reshape(len(people), -1)
    arms = {"duration20": durations, "duration20_demo": np.column_stack((durations, demo)), "raw60": raw60, "raw60_demo": np.column_stack((raw60, demo))}
    if not np.isfinite(raw60).all():
        raise ValueError("Nonfinite raw features")
    fold_rows, predictions = [], []
    for fold_no, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(people, y), 1):
        row = {"fold": fold_no, "models": {}}
        for arm, features in arms.items():
            for name in ("logistic", "rf", "xgb"):
                model = create_model(name)
                model.fit(features[train], y[train])
                score = model.decision_function(features[test]) if name == "logistic" else model.predict_proba(features[test])[:, 1]
                key = f"{arm}_{name}"
                row["models"][key] = {
                    "auroc": float(roc_auc_score(y[test], score)),
                    "balanced_accuracy": float(balanced_accuracy_score(y[test], score >= (0 if name == "logistic" else .5))),
                }
                predictions.extend((fold_no, sid, int(label), key, float(value)) for sid, label, value in zip(people[test], y[test], score))
        fold_rows.append(row)
        print(f"Finished raw trial outer fold {fold_no}/5", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "outer_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("fold", "subject_id", "group", "model", "score"))
        writer.writerows(predictions)
    for name in ("rf", "xgb"):
        final = create_model(name).fit(durations, y)
        joblib.dump(final, out_dir / f"final_duration20_{name}.joblib")
    summary = {
        key: {
            "mean_fold_auroc": float(np.mean([f["models"][key]["auroc"] for f in fold_rows])),
            "mean_fold_balanced_accuracy": float(np.mean([f["models"][key]["balanced_accuracy"] for f in fold_rows])),
            "fold_aurocs": [f["models"][key]["auroc"] for f in fold_rows],
        }
        for key in fold_rows[0]["models"]
    }
    report = {
        "protocol": "Fixed regularized models, same 5 participant-held-out folds; 20 ordered raw per-trial values per channel",
        "feature_source": "Raw trial durations, raw coordinate path lengths, and button press counts; no Mouse2Vec input or embedding aggregation",
        "n_participants": len(people), "n_features_raw60": 60,
        "trial_audit_sha256": sha256(path), "demographics_sha256": sha256(DEMOGRAPHICS),
        "model_settings": {"logistic": "L2 C=0.1", "rf": "300 trees, depth 3, >=8 people per leaf", "xgb": "150 trees, depth 2, min_child_weight 2, gamma 0.5, lambda 10, alpha 1"},
        "final_models": "final_duration20_rf.joblib and final_duration20_xgb.joblib fitted on all 74 real participants after CV; exploratory candidates, not externally validated",
        "summary": summary, "folds": fold_rows,
        "caution": "Exploratory repeated use of the same cohort. Trial timing and click counts can be task-specific and should not be assumed present in ordinary mouse use.",
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    evaluate()
