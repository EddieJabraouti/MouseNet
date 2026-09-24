"""Fixed regularized baselines on the supplied TMT processed digital features.

These are an alternate representation of the raw TMT recording, not Mouse2Vec
embeddings. All features belong to one participant; every outer fold holds
out participants and fits preprocessing on its training participants only.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .model import DEMOGRAPHICS, ROOT, SEED, demographic_features, sha256


DATA = ROOT / "data/nm9xy-osfstorage-processed_data-archive/df_digital_tmt_with_target.csv"
OUT = ROOT / "exp1/runs/processed_feature_baselines"


def estimator(name: str):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight="balanced", solver="liblinear"))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=8, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=4)
    if name == "xgb":
        return XGBClassifier(n_estimators=100, learning_rate=.03, max_depth=1, min_child_weight=5, gamma=2, reg_lambda=25, reg_alpha=2, subsample=.8, colsample_bytree=.8, tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=4)
    if name == "xgb_moderate":
        return XGBClassifier(n_estimators=150, learning_rate=.03, max_depth=2, min_child_weight=2, gamma=.5, reg_lambda=10, reg_alpha=1, subsample=.8, colsample_bytree=.8, tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=4)
    raise ValueError(name)


def evaluate(out_dir: Path = OUT):
    # macOS Accelerate may warn in sklearn's small finite matmuls; all inputs
    # and every resulting held-out score are checked for finiteness below.
    warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning, module="sklearn.utils.extmath")
    data = pd.read_csv(DATA, dtype={"subject_id": str})
    if len(data) != 74 or data.subject_id.nunique() != 74:
        raise ValueError("Processed cohort ID audit failed")
    # Match np.unique() lexical ID order in the established Mouse2Vec folds.
    data = data.sort_values("subject_id").reset_index(drop=True)
    people = data.subject_id.to_numpy()
    y = data.group.to_numpy(dtype=int)
    mouse_columns = [column for column in data if column not in ("subject_id", "group")]
    mouse = data[mouse_columns].to_numpy(dtype=np.float64)
    demo = demographic_features(people, y)
    if mouse.shape != (74, 103) or not np.isfinite(mouse).all() or set(y) != {0, 1}:
        raise ValueError("Processed feature audit failed")
    arms = {"digital": mouse, "demographics": demo, "combined": np.column_stack((mouse, demo))}
    predictions = []
    fold_rows = []
    for fold_no, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(people, y), 1):
        if set(people[train]) & set(people[test]):
            raise AssertionError("Participant leakage")
        row = {"fold": fold_no, "models": {}}
        for arm, X in arms.items():
            for name in ("logistic", "rf", "xgb", "xgb_moderate"):
                model = estimator(name)
                model.fit(X[train], y[train])
                scores = model.decision_function(X[test]) if name == "logistic" else model.predict_proba(X[test])[:, 1]
                if not np.isfinite(scores).all():
                    raise ValueError("Nonfinite scores")
                key = f"{arm}_{name}"
                row["models"][key] = {
                    "auroc": float(roc_auc_score(y[test], scores)),
                    "balanced_accuracy": float(balanced_accuracy_score(y[test], scores >= (0 if name == "logistic" else .5))),
                }
                predictions.extend((fold_no, sid, int(group), key, float(score)) for sid, group, score in zip(people[test], y[test], scores))
        fold_rows.append(row)
        print(f"Finished processed-feature outer fold {fold_no}/5", flush=True)
    summary = {
        key: {
            "mean_fold_auroc": float(np.mean([fold["models"][key]["auroc"] for fold in fold_rows])),
            "mean_fold_balanced_accuracy": float(np.mean([fold["models"][key]["balanced_accuracy"] for fold in fold_rows])),
            "fold_aurocs": [fold["models"][key]["auroc"] for fold in fold_rows],
        }
        for key in fold_rows[0]["models"]
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "outer_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("fold", "subject_id", "group", "model", "score"))
        writer.writerows(predictions)
    report = {
        "protocol": "Fixed regularized classifiers; stratified 5-fold outer validation by participant; train-fold-only scaling for logistic regression",
        "feature_source": "103 supplied processed digital TMT variables computed per participant; demographic variables are sex, age, education",
        "n_participants": 74, "n_digital_features": len(mouse_columns),
        "processed_data_sha256": sha256(DATA), "demographics_sha256": sha256(DEMOGRAPHICS),
        "model_settings": {"logistic": "L2 C=0.1", "rf": "300 trees, max depth 3, >=8 participants per leaf", "xgb": "100 trees, depth 1, min_child_weight 5, gamma 2, lambda 25, alpha 2", "xgb_moderate": "150 trees, depth 2, min_child_weight 2, gamma 0.5, lambda 10, alpha 1"},
        "numeric_note": "Finite inputs and predictions checked; spurious macOS Accelerate matmul warnings filtered as in the earlier experiment.",
        "summary": summary, "folds": fold_rows,
        "caution": "Exploratory reuse of a cohort and split already examined in other experiments; no external test. This representation is task-specific and may not transfer to ordinary mouse use.",
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    evaluate()
