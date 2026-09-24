"""Validate saved synthetic TMT trajectories against their real sources."""

from __future__ import annotations

import csv
import json

import numpy as np

from .audit_preprocessing import OUT as AUDIT_DIR
from .model import ROOT, demographic_features, demographic_targets
from .synthetic_tmt import OUT as SYNTH_DIR


def audit():
    with (AUDIT_DIR / "trial_audit.csv").open(newline="") as handle:
        source_trials = {(r["subject_id"], int(r["trial_order"])): r for r in csv.DictReader(handle)}
    targets = demographic_targets()
    people = np.array(sorted(targets))
    y = np.array([targets[sid] for sid in people])
    source_demo = dict(zip(people, demographic_features(people, y)))
    with np.load(SYNTH_DIR / "synthetic_raw_trajectories.npz", allow_pickle=False) as data:
        t = data["timestamps_s"]
        xy = data["xy_normalized"]
        buttons = data["left_mid_right_button"]
        offsets = data["trial_offsets"]
        synth_id = data["synthetic_subject_id"]
        source = data["source_subject_id"]
        group = data["group"]
        demo = data["demographics"]
        order = data["trial_order"]
    duration_ratio, path_ratio = [], []
    for i in range(len(synth_id)):
        sid = source[i]
        if group[i] != targets[sid] or demo[i, 0] != source_demo[sid][0]:
            raise ValueError("Synthetic source label/sex mismatch")
        if not (55 <= demo[i, 1] <= 90 and abs(demo[i, 1] - source_demo[sid][1]) <= 2 and 7 <= demo[i, 2] <= 20 and abs(demo[i, 2] - source_demo[sid][2]) <= 1):
            raise ValueError("Synthetic demographic bounds violated")
        start, stop = offsets[i:i + 2]
        trial_time = t[start:stop]
        trial_xy = xy[start:stop]
        if len(trial_time) < 2 or np.any(np.diff(trial_time) <= 0) or np.any(trial_xy < 0) or np.any(trial_xy > 1):
            raise ValueError("Synthetic time/coordinate bounds violated")
        original = source_trials[(sid, int(order[i]))]
        duration_ratio.append(float((trial_time[-1] - trial_time[0]) / float(original["duration_s"])))
        height_xy = np.column_stack((trial_xy[:, 0] * 16 / 9 - 8 / 9, .5 - trial_xy[:, 1]))
        length = float(np.linalg.norm(np.diff(height_xy, axis=0), axis=1).sum())
        path_ratio.append(length / float(original["raw_path"]))
        press = np.zeros(len(trial_time), dtype=bool)
        for channel in range(3):
            b = buttons[start:stop, channel]
            press |= (b > 0) & np.r_[True, b[:-1] == 0]
        if int(press.sum()) != int(original["clicks"]):
            raise ValueError("Synthetic click count differs from source")
    duration_ratio = np.asarray(duration_ratio)
    path_ratio = np.asarray(path_ratio)
    if len(synth_id) != 2960 or duration_ratio.min() < .979 or duration_ratio.max() > 1.021:
        raise ValueError("Synthetic trial count/duration bounds invalid")
    report = {
        "n_synthetic_trials": len(synth_id), "n_synthetic_profiles": len(set(synth_id)),
        "all_demographics_within_bounds": True, "all_labels_and_sex_match_source": True,
        "all_timestamps_increasing": True, "all_coordinates_in_screen": True,
        "all_click_counts_match_source": True,
        "duration_ratio_min_median_max": [float(np.min(duration_ratio)), float(np.median(duration_ratio)), float(np.max(duration_ratio))],
        "path_ratio_min_median_max": [float(np.min(path_ratio)), float(np.median(path_ratio)), float(np.max(path_ratio))],
        "warning": "Small perturbations preserve mechanics, but plausible-looking synthetic data do not create independent cognitive labels or guarantee clinical fidelity.",
    }
    (SYNTH_DIR / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    audit()
