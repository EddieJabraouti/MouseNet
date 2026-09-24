"""Fixed, heavily regularized tree comparisons on frozen Mouse2Vec windows.

Every validation fold contains only real, unseen participants. Synthetic
descendants, when provided, enter only the source participant's training fold.
No embedding averaging takes place; window scores are aggregated afterward.
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .audit_preprocessing import OUT as AUDIT_DIR
from .model import DEMOGRAPHICS, ROOT, RUN_DIR, SEED, demographic_features, demographic_targets, sha256


OUT = ROOT / "exp1/runs/regularized_trees"
SPECS = (("svm_pls2", 2), ("rf_pls2", 2), ("rf_pls8", 8), ("xgb_pls2", 2), ("xgb_pls8", 8))


def read_real(path: Path):
    with np.load(path, allow_pickle=False) as data:
        z = data["embeddings"].astype(np.float64)
        ids = data["subject_id"]
        trial_order = data["trial_order"]
    targets = demographic_targets()
    people = np.unique(ids)
    if len(people) != 74 or len(z) != 26655 or set(people) != set(targets) or z.shape[1] != 128:
        raise ValueError("Real embedding cohort audit failed")
    if not np.isfinite(z).all() or len(trial_order) != len(z):
        raise ValueError("Nonfinite or misaligned real embeddings")
    y = np.asarray([targets[sid] for sid in people], dtype=int)
    demo = demographic_features(people, y)
    by_id = {sid: (y[i], demo[i]) for i, sid in enumerate(people)}
    return z, ids, trial_order, people, y, demo, by_id


def duration_for_windows(ids, trial_order):
    with (AUDIT_DIR / "trial_audit.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["subject_id"], int(row["trial_order"])): float(row["duration_s"]) for row in rows}
    if len(lookup) != 1480:
        raise ValueError("Trial duration audit incomplete")
    return np.asarray([lookup[(sid, int(trial))] for sid, trial in zip(ids, trial_order)])


def read_synthetic(path: Path | None):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as data:
        z = data["embeddings"].astype(np.float64)
        source = data["source_subject_id"]
        synth_id = data["synthetic_subject_id"]
        y = data["group"].astype(int)
        demo = data["demographics"].astype(np.float64)
    if z.ndim != 2 or z.shape[1] != 128 or len(z) != len(source) or len(z) != len(synth_id) or demo.shape != (len(z), 3):
        raise ValueError("Synthetic embedding dimensions invalid")
    if not np.isfinite(z).all() or not np.isfinite(demo).all():
        raise ValueError("Nonfinite synthetic data")
    return z, source, synth_id, y, demo


def balanced_lineage_weights(ids, source, labels_by_id, real_ids):
    """One participant lineage gets equal total weight, even when augmented."""
    lineage = np.concatenate((ids, source))
    counts = Counter(lineage)
    class_people = Counter(labels_by_id[sid] for sid in real_ids)
    weights = np.array([1 / (2 * class_people[labels_by_id[sid]] * counts[sid]) for sid in lineage])
    return weights * len(ids) / weights.sum()


def create_model(name: str, row_multiplier: float = 1):
    if name.startswith("rf"):
        return RandomForestClassifier(
            n_estimators=160, max_depth=4, min_samples_leaf=int(np.ceil(250 * row_multiplier)),
            max_features="sqrt", bootstrap=True, random_state=SEED, n_jobs=4,
        )
    if name.startswith("xgb"):
        return XGBClassifier(
            n_estimators=120, learning_rate=.03, max_depth=2,
            min_child_weight=100, gamma=5, reg_lambda=50, reg_alpha=2,
            subsample=.8, colsample_bytree=.8, max_bin=128,
            tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=4,
        )
    if name.startswith("svm"):
        return SVC(kernel="rbf", C=1, gamma="scale", cache_size=500)
    raise ValueError(name)


def participant_scores(model, X, ids, people):
    if isinstance(model, SVC):
        scores = model.decision_function(X)
    else:
        scores = model.predict_proba(X)[:, 1]
    return np.asarray([np.median(scores[ids == sid]) for sid in people])


def evaluate(real_path: Path, out_dir: Path, synthetic_path: Path | None = None, include_direct: bool = False, include_duration: bool = False):
    z, ids, trial_order, people, y, demo, by_id = read_real(real_path)
    synthetic = read_synthetic(synthetic_path)
    if include_duration and synthetic is not None:
        raise ValueError("Duration side channel is currently evaluated without synthetic augmentation")
    durations = duration_for_windows(ids, trial_order) if include_duration else None
    if synthetic is not None:
        sz, source, synth_ids, sy, sdemo = synthetic
        if not set(source).issubset(set(people)) or set(synth_ids) & set(people):
            raise ValueError("Synthetic ID/source overlap")
        if any(sy[i] != by_id[sid][0] for i, sid in enumerate(source)):
            raise ValueError("Synthetic class differs from source")
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(people, y))
    predictions = []
    fold_results = []
    for fold_no, (train, test) in enumerate(folds, 1):
        train_ids, test_ids = people[train], people[test]
        real_train = np.isin(ids, train_ids)
        real_test = np.isin(ids, test_ids)
        train_z, train_ids_window = z[real_train], ids[real_train]
        train_demo = np.asarray([by_id[sid][1] for sid in train_ids_window])
        train_y = np.asarray([by_id[sid][0] for sid in train_ids_window])
        source_added = np.array([], dtype=ids.dtype)
        if synthetic is not None:
            allowed = np.isin(source, train_ids)
            if np.any(np.isin(source[allowed], test_ids)):
                raise AssertionError("Synthetic descendant of held-out participant")
            source_added = source[allowed]
            train_z = np.concatenate((train_z, sz[allowed]))
            train_ids_window = np.concatenate((train_ids_window, synth_ids[allowed]))
            train_demo = np.concatenate((train_demo, sdemo[allowed]))
            train_y = np.concatenate((train_y, sy[allowed]))
        test_z = z[real_test]
        test_ids_window = ids[real_test]
        test_demo = np.asarray([by_id[sid][1] for sid in test_ids_window])
        weights = balanced_lineage_weights(ids[real_train], source_added, {sid: by_id[sid][0] for sid in train_ids}, train_ids)
        scaler = StandardScaler().fit(train_z)
        train_scaled = scaler.transform(train_z)
        test_scaled = scaler.transform(test_z)
        demo_scaler = StandardScaler().fit(demo[train])
        dtrain, dtest = demo_scaler.transform(train_demo), demo_scaler.transform(test_demo)
        if include_duration:
            duration_scaler = StandardScaler().fit(durations[real_train, None])
            duration_train = duration_scaler.transform(durations[real_train, None])
            duration_test = duration_scaler.transform(durations[real_test, None])
        fold_row = {"fold": fold_no, "n_train_real_participants": len(train), "n_train_synthetic_windows": len(source_added), "models": {}}
        for components in (2, 8):
            pls = PLSRegression(n_components=components, scale=False).fit(train_scaled, train_y)
            pt = pls.transform(train_scaled)
            pv = pls.transform(test_scaled)
            ps = StandardScaler().fit(pt)
            pt, pv = ps.transform(pt) / np.sqrt(components), ps.transform(pv) / np.sqrt(components)
            for name, n_comp in SPECS:
                if n_comp != components:
                    continue
                arms = ("mouse", "combined", "mouse_duration", "combined_duration") if include_duration else ("mouse", "combined")
                for arm in arms:
                    Xt = pt if not arm.startswith("combined") else np.column_stack((pt, dtrain / np.sqrt(3)))
                    Xv = pv if not arm.startswith("combined") else np.column_stack((pv, dtest / np.sqrt(3)))
                    if arm.endswith("duration"):
                        Xt = np.column_stack((Xt, duration_train))
                        Xv = np.column_stack((Xv, duration_test))
                    model = create_model(name, len(train_z) / int(real_train.sum()))
                    model.fit(Xt, train_y, sample_weight=weights)
                    score = participant_scores(model, Xv, test_ids_window, test_ids)
                    key = f"{name}_{arm}"
                    fold_row["models"][key] = {
                        "auroc": float(roc_auc_score(y[test], score)),
                        "balanced_accuracy": float(balanced_accuracy_score(y[test], score >= (0 if isinstance(model, SVC) else .5))),
                    }
                    predictions.extend((fold_no, sid, int(label), key, float(value)) for sid, label, value in zip(test_ids, y[test], score))
        if include_direct:
            for name in ("rf_direct", "xgb_direct"):
                for arm in ("mouse", "combined"):
                    Xt = train_scaled if arm == "mouse" else np.column_stack((train_scaled, dtrain))
                    Xv = test_scaled if arm == "mouse" else np.column_stack((test_scaled, dtest))
                    model = create_model(name, len(train_z) / int(real_train.sum()))
                    model.fit(Xt, train_y, sample_weight=weights)
                    score = participant_scores(model, Xv, test_ids_window, test_ids)
                    key = f"{name}_{arm}"
                    fold_row["models"][key] = {
                        "auroc": float(roc_auc_score(y[test], score)),
                        "balanced_accuracy": float(balanced_accuracy_score(y[test], score >= .5)),
                    }
                    predictions.extend((fold_no, sid, int(label), key, float(value)) for sid, label, value in zip(test_ids, y[test], score))
        fold_results.append(fold_row)
        print(f"Finished tree outer fold {fold_no}/5", flush=True)
    with (out_dir / "outer_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("fold", "subject_id", "group", "model", "score"))
        writer.writerows(predictions)
    summary = {
        key: {
            "mean_fold_auroc": float(np.mean([fold["models"][key]["auroc"] for fold in fold_results])),
            "mean_fold_balanced_accuracy": float(np.mean([fold["models"][key]["balanced_accuracy"] for fold in fold_results])),
            "fold_aurocs": [fold["models"][key]["auroc"] for fold in fold_results],
        }
        for key in fold_results[0]["models"]
    }
    report = {
        "protocol": "Fixed heavily regularized tree/SVM recipes; 5 stratified outer folds by real participant; training-fold-only PLS/scaling; synthetic descendants restricted to training source folds",
        "primary_metric": "mean held-out participant-fold AUROC; median applied only to post-classifier window scores",
        "n_real_participants": len(people), "n_real_windows": len(z),
        "synthetic_training": synthetic_path is not None,
        "direct_128d_trees_included": include_direct,
        "trial_duration_side_channel_included": include_duration,
        "real_embeddings_sha256": sha256(real_path),
        "synthetic_embeddings_sha256": sha256(synthetic_path) if synthetic_path else None,
        "demographics_sha256": sha256(DEMOGRAPHICS),
        "model_settings": {"rf": "160 trees, depth <=4, >=250 real-equivalent windows/leaf, sqrt features", "xgb": "120 trees, depth <=2, min_child_weight 100, gamma 5, lambda 50, alpha 2, subsample/colsample 0.8", "svm": "RBF C=1 gamma=scale"},
        "summary": summary, "folds": fold_results,
        "caution": "Exploratory repeated use of this 74-person cohort. Synthetic examples are descendants of real people and never count as independent test participants.",
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=RUN_DIR / "window_embeddings.npz")
    parser.add_argument("--synthetic", type=Path)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--include-direct", action="store_true", help="Also fit RF/XGBoost directly on all 128 embedding dimensions")
    parser.add_argument("--include-duration", action="store_true", help="Also evaluate trial-duration side channels after PLS")
    args = parser.parse_args()
    evaluate(args.real, args.output, args.synthetic, args.include_direct, args.include_duration)
