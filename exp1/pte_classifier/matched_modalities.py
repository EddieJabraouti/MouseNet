"""Matched, participant-held-out ablation of mouse, demographics, and TMT time.

All arms use identical folds, model recipes, and per-participant training
weights. Mouse2Vec windows remain separate through training; only the
classifier's window scores are reduced to one score per held-out person.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .model import DEMOGRAPHICS, ROOT, sha256
from .synthetic_raw_eval import real_profiles
from .tree_experiment import read_real


EMBEDDINGS = ROOT / "exp1/runs/representation_variants/hold_window_fft.npz"
TRIAL_AUDIT = ROOT / "exp1/runs/preprocessing_audit/trial_audit.csv"
OUT = ROOT / "exp1/runs/matched_modalities"
SEEDS = tuple(range(42, 52))
ARMS = {
    "mouse": ("mouse",),
    "demographic": ("demographic",),
    "time": ("time",),
    "combined": ("mouse", "demographic"),
    "mouse_time": ("mouse", "time"),
    "demographic_time": ("demographic", "time"),
    "all": ("mouse", "demographic", "time"),
}


def estimator(name: str, seed: int):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=160, max_depth=4, min_samples_leaf=250,
            max_features="sqrt", bootstrap=True, random_state=seed, n_jobs=4,
        )
    if name == "xgb":
        return XGBClassifier(
            n_estimators=120, learning_rate=.03, max_depth=2,
            min_child_weight=100, gamma=5, reg_lambda=50, reg_alpha=2,
            subsample=.8, colsample_bytree=.8, max_bin=128,
            tree_method="hist", eval_metric="logloss", random_state=seed,
            n_jobs=4,
        )
    raise ValueError(name)


def run(out_dir: Path = OUT, seeds: tuple[int, ...] = SEEDS, models: tuple[str, ...] = ("rf", "xgb")):
    z, ids, _, people, y, demo, _ = read_real(EMBEDDINGS)
    profile_ids, profile_y, profile_demo, raw = real_profiles()
    if not np.array_equal(people, profile_ids) or not np.array_equal(y, profile_y) or not np.array_equal(demo, profile_demo):
        raise ValueError("Mouse, trial time, and demographic participant alignment failed")
    time = raw[:, :20]
    if time.shape != (74, 20) or not np.isfinite(time).all():
        raise ValueError("Invalid ordered 20-trial time matrix")
    person_row = {sid: i for i, sid in enumerate(people)}
    window_person = np.asarray([person_row[sid] for sid in ids])
    rows = []
    repeats = []
    for seed in seeds:
        fold_results = []
        for fold_no, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(people, y), 1):
            train_mask = np.isin(window_person, train)
            test_mask = np.isin(window_person, test)
            tr_person = window_person[train_mask]
            te_person = window_person[test_mask]
            if set(tr_person) != set(train) or set(te_person) != set(test) or set(train) & set(test):
                raise AssertionError("Participant split leakage or missing windows")

            # One person contributes equal total weight within their class,
            # independent of how many overlapping windows they produced.
            counts = Counter(tr_person)
            class_counts = Counter(y[train])
            weights = np.asarray([
                1 / (2 * class_counts[y[i]] * counts[i]) for i in tr_person
            ])
            weights *= len(tr_person) / weights.sum()

            z_scaler = StandardScaler().fit(z[train_mask])
            pls = PLSRegression(n_components=2, scale=False).fit(
                z_scaler.transform(z[train_mask]), y[tr_person]
            )
            mouse_train = pls.transform(z_scaler.transform(z[train_mask]))
            mouse_test = pls.transform(z_scaler.transform(z[test_mask]))
            mouse_scaler = StandardScaler().fit(mouse_train)
            mouse_train = mouse_scaler.transform(mouse_train) / np.sqrt(2)
            mouse_test = mouse_scaler.transform(mouse_test) / np.sqrt(2)

            demo_scaler = StandardScaler().fit(demo[train])
            time_scaler = StandardScaler().fit(time[train])
            train_blocks = {
                "mouse": mouse_train,
                "demographic": demo_scaler.transform(demo[tr_person]) / np.sqrt(3),
                "time": time_scaler.transform(time[tr_person]) / np.sqrt(20),
            }
            test_blocks = {
                "mouse": mouse_test,
                "demographic": demo_scaler.transform(demo[te_person]) / np.sqrt(3),
                "time": time_scaler.transform(time[te_person]) / np.sqrt(20),
            }
            fold_scores = {}
            for model_name in models:
                for arm, blocks in ARMS.items():
                    x_train = np.column_stack([train_blocks[key] for key in blocks])
                    x_test = np.column_stack([test_blocks[key] for key in blocks])
                    model = estimator(model_name, seed)
                    model.fit(x_train, y[tr_person], sample_weight=weights)
                    window_scores = model.predict_proba(x_test)[:, 1]
                    scores = np.asarray([
                        np.median(window_scores[te_person == i]) for i in test
                    ])
                    auc = float(roc_auc_score(y[test], scores))
                    fold_scores[f"{model_name}_{arm}"] = auc
                    rows.extend((seed, fold_no, sid, int(y[i]), model_name, arm, float(score))
                                for i, sid, score in zip(test, people[test], scores))
            fold_results.append({"fold": fold_no, "held_out_ids": people[test].tolist(), "auroc": fold_scores})
            print(f"Completed seed {seed}, fold {fold_no}/5", flush=True)
        repeats.append({"seed": seed, "folds": fold_results})

    keys = [f"{name}_{arm}" for name in models for arm in ARMS]
    summary = {}
    for key in keys:
        repeat_means = [float(np.mean([f["auroc"][key] for f in repeat["folds"]])) for repeat in repeats]
        summary[key] = {
            "mean_fold_auroc_across_repeats": float(np.mean(repeat_means)),
            "repeat_mean_min": float(np.min(repeat_means)),
            "repeat_mean_max": float(np.max(repeat_means)),
            "seed42_mean_fold_auroc": repeat_means[0] if seeds[0] == 42 else None,
            "repeat_means": repeat_means,
        }
    report = {
        "protocol": "Ten prespecified stratified five-fold participant-held-out shuffles, seeds 42-51; one shared split and fixed classifier recipe per arm; real cohort only; all transforms fit within training participants; equal total training weight per participant within class; median of post-classifier window scores only",
        "feature_definitions": {
            "mouse": "Each five-second corrected frozen Mouse2Vec 128D window, compressed with training-fold-only PLS to 2D; windows are never averaged",
            "demographic": "Sex, age, and years of education, repeated unchanged per participant window",
            "time": "All 20 ordered raw TMT trial completion times, repeated unchanged per participant window",
            "combined": "mouse + demographic",
            "all": "mouse + demographic + time",
        },
        "n_participants": len(people),
        "class_counts": {str(k): int(v) for k, v in Counter(y).items()},
        "n_windows": len(z),
        "models": list(models),
        "model_settings": {
            "rf": "160 trees, depth <=4, >=250 windows/leaf, sqrt feature sampling",
            "xgb": "120 trees, depth <=2, min_child_weight 100, gamma 5, lambda 50, alpha 2, subsample and colsample 0.8",
        },
        "embeddings_sha256": sha256(EMBEDDINGS),
        "trial_audit_sha256": sha256(TRIAL_AUDIT),
        "demographics_sha256": sha256(DEMOGRAPHICS),
        "summary": summary,
        "repeats": repeats,
        "caution": "Repeated folds reuse the same 74 people. This is an exploratory matched ablation after prior cohort-level inspection, not an independent test set or an external performance estimate. Trial completion times are specific to TMT.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "outer_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("seed", "fold", "subject_id", "group", "model", "arm", "score"))
        writer.writerows(rows)
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--models", choices=("rf", "xgb"), nargs="+", default=("rf", "xgb"))
    args = parser.parse_args()
    run(args.output, tuple(args.seeds), tuple(args.models))
