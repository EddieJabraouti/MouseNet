"""Evaluate whether synthetic raw TMT descendants help regularized RF models."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .audit_preprocessing import OUT as AUDIT_DIR
from .model import DEMOGRAPHICS, ROOT, SEED, demographic_features, demographic_targets, sha256
from .synthetic_tmt import OUT as SYNTH_DIR


OUT = ROOT / "exp1/runs/synthetic_raw_eval"


def real_profiles():
    with (AUDIT_DIR / "trial_audit.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {}
    for row in rows:
        by_id.setdefault(row["subject_id"], []).append(row)
    labels = demographic_targets()
    people = np.array(sorted(labels))
    y = np.array([labels[sid] for sid in people])
    demo = demographic_features(people, y)
    X = []
    for sid in people:
        trials = sorted(by_id[sid], key=lambda row: int(row["trial_order"]))
        if len(trials) != 20:
            raise ValueError("Missing real trials")
        X.append([[float(row[field]) for row in trials] for field in ("duration_s", "raw_path", "clicks")])
    return people, y, demo, np.asarray(X).reshape(len(people), 60)


def synthetic_profiles(path: Path):
    with np.load(path, allow_pickle=False) as data:
        t = data["timestamps_s"]
        xy = data["xy_normalized"]
        buttons = data["left_mid_right_button"]
        offsets = data["trial_offsets"]
        synth_ids = data["synthetic_subject_id"]
        sources = data["source_subject_id"]
        labels = data["group"]
        demos = data["demographics"]
        orders = data["trial_order"]
    profiles = {}
    for i, sid in enumerate(synth_ids):
        start, stop = offsets[i:i + 2]
        time = t[start:stop]
        coords = xy[start:stop]
        xy_height = np.column_stack((coords[:, 0] * 16 / 9 - 8 / 9, .5 - coords[:, 1]))
        path_length = float(np.linalg.norm(np.diff(xy_height, axis=0), axis=1).sum())
        button = buttons[start:stop]
        presses = np.zeros(len(button), dtype=bool)
        for axis in range(3):
            presses |= (button[:, axis] > 0) & np.r_[True, button[:-1, axis] == 0]
        clicks = int(presses.sum())
        entry = profiles.setdefault(sid, {"source": sources[i], "group": int(labels[i]), "demographics": demos[i], "trials": {}})
        if entry["source"] != sources[i] or entry["group"] != int(labels[i]) or not np.array_equal(entry["demographics"], demos[i]):
            raise ValueError("Synthetic profile metadata mismatch")
        entry["trials"][int(orders[i])] = (float(time[-1] - time[0]), path_length, clicks)
    ids = sorted(profiles)
    source = np.array([profiles[sid]["source"] for sid in ids])
    y = np.array([profiles[sid]["group"] for sid in ids])
    demo = np.stack([profiles[sid]["demographics"] for sid in ids])
    X = []
    for sid in ids:
        trials = profiles[sid]["trials"]
        if set(trials) != set(range(20)):
            raise ValueError("Incomplete synthetic profile")
        X.append([[trials[i][channel] for i in range(20)] for channel in range(3)])
    return np.array(ids), source, y, demo, np.asarray(X).reshape(len(ids), 60)


def evaluate(out_dir: Path = OUT):
    people, y, demo, X = real_profiles()
    synth_path = SYNTH_DIR / "synthetic_raw_trajectories.npz"
    synth_ids, source, sy, sdemo, sX = synthetic_profiles(synth_path)
    if len(people) != 74 or len(synth_ids) != 148 or len(set(synth_ids)) != 148:
        raise ValueError("Profile counts invalid")
    labels_by_id = dict(zip(people, y))
    if any(sy[i] != labels_by_id[sid] for i, sid in enumerate(source)):
        raise ValueError("Synthetic source class mismatch")
    fold_results, predictions = [], []
    for fold_no, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(people, y), 1):
        allowed = np.isin(source, people[train])
        if np.any(np.isin(source[allowed], people[test])):
            raise AssertionError("Descendant leakage")
        fold = {"fold": fold_no, "models": {}}
        for arm in ("duration20", "raw60", "raw60_demo"):
            real = X[:, :20] if arm == "duration20" else X
            synthetic = sX[:, :20] if arm == "duration20" else sX
            if arm == "raw60_demo":
                real = np.column_stack((real, demo))
                synthetic = np.column_stack((synthetic, sdemo))
            x_train = np.concatenate((real[train], synthetic[allowed]))
            y_train = np.concatenate((y[train], sy[allowed]))
            lineage = np.concatenate((people[train], source[allowed]))
            counts = Counter(lineage)
            class_counts = Counter(y[train])
            weights = np.array([1 / (2 * class_counts[labels_by_id[sid]] * counts[sid]) for sid in lineage])
            weights *= len(train) / weights.sum()
            model = RandomForestClassifier(
                n_estimators=300, max_depth=3,
                min_samples_leaf=int(np.ceil(8 * len(x_train) / len(train))),
                max_features="sqrt", random_state=SEED, n_jobs=4,
            )
            model.fit(x_train, y_train, sample_weight=weights)
            scores = model.predict_proba(real[test])[:, 1]
            fold["models"][arm] = {
                "auroc": float(roc_auc_score(y[test], scores)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test], scores >= .5)),
            }
            predictions.extend((fold_no, sid, int(label), arm, float(score)) for sid, label, score in zip(people[test], y[test], scores))
        fold_results.append(fold)
        print(f"Finished synthetic raw outer fold {fold_no}/5", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "outer_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("fold", "subject_id", "group", "arm", "score"))
        writer.writerows(predictions)
    summary = {
        arm: {
            "mean_fold_auroc": float(np.mean([fold["models"][arm]["auroc"] for fold in fold_results])),
            "mean_fold_balanced_accuracy": float(np.mean([fold["models"][arm]["balanced_accuracy"] for fold in fold_results])),
            "fold_aurocs": [fold["models"][arm]["auroc"] for fold in fold_results],
        }
        for arm in fold_results[0]["models"]
    }
    report = {
        "protocol": "Same 5 real-participant outer folds; 2 synthetic descendants per training source only; every real source lineage equal total sample weight; RF leaf size scaled by replica count",
        "n_real_participants": len(people), "n_synthetic_profiles": len(synth_ids),
        "synthetic_raw_sha256": sha256(synth_path), "demographics_sha256": sha256(DEMOGRAPHICS),
        "summary": summary, "folds": fold_results,
        "caution": "Only real unseen participants are scored. Synthetic descendants are not independent observations or independent labels.",
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    evaluate()
