"""Audit information retained by the existing raw TMT -> Mouse2Vec adapter."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .model import RAW_DIR, ROOT, SAMPLE_HZ, demographic_targets, parse_array


OUT = ROOT / "exp1/runs/preprocessing_audit"


def path_length(xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def summarize(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(("min", "median", "p90", "max"), np.quantile(values, (0, .5, .9, 1)))}


def audit(out_dir: Path = OUT):
    csv.field_size_limit(10_000_000)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = demographic_targets()
    rows = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        sid = path.name.split("_", 1)[0]
        with path.open(newline="", encoding="utf-8-sig") as handle:
            trials = list(csv.DictReader(handle))
        trial_order = 0
        for row in trials:
            if row.get("blocks.thisIndex") != "1":
                continue
            t = parse_array(row["trialMouse.time"], "time")
            x = parse_array(row["trialMouse.x"], "x")
            y = parse_array(row["trialMouse.y"], "y")
            xy = np.column_stack((x, y))
            raw_dt = np.diff(t)
            grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * SAMPLE_HZ)) + 1) / SAMPLE_HZ
            prior = np.maximum(np.searchsorted(t, grid, side="right") - 1, 0)
            held = xy[prior]
            interp = np.column_stack((np.interp(grid, t, x), np.interp(grid, t, y)))
            buttons = [parse_array(row[f"trialMouse.{name}Button"], name) for name in ("left", "mid", "right")]
            presses = np.zeros(len(t), dtype=bool)
            for b in buttons:
                presses |= (b > 0) & np.r_[True, b[:-1] == 0]
            raw_length = path_length(xy)
            held_length = path_length(held)
            interp_length = path_length(interp)
            rows.append({
                "subject_id": sid, "group": labels[sid], "trial_order": trial_order,
                "trial_type": row["trial_type"], "duration_s": float(t[-1] - t[0]),
                "raw_samples": len(t), "raw_median_hz": float(1 / np.median(raw_dt)),
                "raw_path": raw_length, "hold_path": held_length, "linear_path": interp_length,
                "hold_path_fraction": held_length / raw_length if raw_length else 1,
                "linear_path_fraction": interp_length / raw_length if raw_length else 1,
                "hold_stationary_fraction": float(np.mean(np.linalg.norm(np.diff(held, axis=0), axis=1) == 0)),
                "raw_stationary_fraction": float(np.mean(np.linalg.norm(np.diff(xy, axis=0), axis=1) == 0)),
                "clicks": int(presses.sum()),
                "unrepresented_tail_s": float((t[-1] - t[0]) - (grid[-1] - grid[0])),
                "window_count": max(0, (len(grid) - 100) // 20 + 1),
            })
            trial_order += 1
        if trial_order != 20:
            raise ValueError(f"Expected 20 trials for {sid}")
    if len(rows) != 1480:
        raise ValueError("Expected 1480 experimental trials")
    with (out_dir / "trial_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["subject_id"]].append(row)
    people = sorted(by_id, key=int)
    y = np.array([labels[sid] for sid in people])
    aucs = {}
    for part in ("A", "B"):
        for field in ("duration_s", "raw_path", "hold_path", "linear_path", "clicks"):
            values = np.array([np.median([r[field] for r in by_id[sid] if r["trial_type"] == part]) for sid in people])
            aucs[f"{part}_{field}"] = float(roc_auc_score(y, values))
    report = {
        "n_participants": len(people), "n_trials": len(rows),
        "raw_sample_rate_hz": summarize(np.array([r["raw_median_hz"] for r in rows])),
        "raw_sample_count": summarize(np.array([r["raw_samples"] for r in rows])),
        "trial_duration_s": summarize(np.array([r["duration_s"] for r in rows])),
        "hold_path_fraction": summarize(np.array([r["hold_path_fraction"] for r in rows])),
        "linear_path_fraction": summarize(np.array([r["linear_path_fraction"] for r in rows])),
        "raw_stationary_fraction": summarize(np.array([r["raw_stationary_fraction"] for r in rows])),
        "hold_stationary_fraction": summarize(np.array([r["hold_stationary_fraction"] for r in rows])),
        "clicks": summarize(np.array([r["clicks"] for r in rows])),
        "trials_with_clicks": int(sum(r["clicks"] > 0 for r in rows)),
        "path_spearman_raw_vs_hold": float(spearmanr([r["raw_path"] for r in rows], [r["hold_path"] for r in rows]).statistic),
        "path_spearman_raw_vs_linear": float(spearmanr([r["raw_path"] for r in rows], [r["linear_path"] for r in rows]).statistic),
        "descriptive_participant_auc_from_trial_medians": aucs,
        "interpretation": "Descriptive marginal audits on one cohort; not held-out model performance. Path differences quantify information discarded by 20 Hz resampling. Trial duration and click counts are not direct Mouse2Vec encoder inputs.",
    }
    (out_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    audit()
