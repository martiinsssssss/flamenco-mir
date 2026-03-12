#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voicing evaluation for the Cante2MIDI (C2M) dataset using mir_eval.

Repository layout assumed from the user's repo:

FLAMENCO-MIR/
│
├── data/
│   ├── cante2midi_groundTruth/
│   ├── cante2midi_f0/
│   ├── cante2midi_automaticTranscription/
│   ├── cante2midi_audio/
│   └── cante2midi_voicing_predictions/   <-- recommended folder for model outputs
│
└── SourceSeparation/
    └── evaluate_voicing_mir_eval.py      <-- this script

What this script does:
1) Loads frame-wise voicing predictions from your model (.csv or .npy files).
2) Loads note interval annotations from C2M ground truth (.csv with onset,offset).
3) Converts note intervals into frame-wise voiced/unvoiced ground truth.
4) Sweeps a range of thresholds over model activations.
5) Uses mir_eval for voicing recall and voicing false alarm.
6) Computes Kroher & Gómez style:
      - Voicing Precision (Pr-V)
      - Voicing Recall (Rec-V)
      - Voicing F-measure (FM-V)
7) Finds the threshold with the best FM-V.
8) Compares against the reported C2M benchmark:
      Pr-V = 0.92
      Rec-V = 0.94
      FM-V = 0.92
9) Saves:
      - dataset-level threshold sweep CSV
      - per-track metrics at best threshold
      - threshold sweep plot

Dependencies:
    pip install numpy pandas matplotlib mir_eval

Typical usage from repo root:
    python SourceSeparation/evaluate_voicing_mir_eval.py \
        --pred-dir data/cante2midi_voicing_predictions \
        --gt-dir data/cante2midi_groundTruth \
        --hop-size 0.01

Expected prediction format (recommended):
    data/cante2midi_voicing_predictions/<track_id>.csv
    where the 4th column (index 3) is binary voicing:
        1 = voiced, 0 = unvoiced

Also supported for backward compatibility:
    data/cante2midi_voicing_predictions/<track_id>.npy

Expected annotation format:
    data/cante2midi_groundTruth/<track_id>.csv
    with columns:
        onset,offset

Output files default to:
    SourceSeparation/results/voicing_threshold_results.csv
    SourceSeparation/results/per_track_best_threshold.csv
    SourceSeparation/results/voicing_threshold_sweep.png
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mir_eval


# ============================================================
# Data container
# ============================================================

@dataclass
class TrackData:
    track_id: str
    activations: np.ndarray   # shape: (n_frames,), float in [0, 1]
    gt_voiced: np.ndarray     # shape: (n_frames,), bool
    voiced_ratio: float       # gt voiced fraction for this track


# ============================================================
# Utility functions
# ============================================================

def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def harmonic_f1(p: float, r: float) -> float:
    return safe_div(2.0 * p * r, p + r)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# File loading
# ============================================================

def load_predictions(path: Path) -> np.ndarray:
    """
    Load model voicing predictions.

    Supported formats:
    - .csv: 4th column (index 3) is expected to be binary 0/1 voicing
    - .npy: 1D activation array in [0, 1] (backward compatibility)

    Expected shape: (num_frames,)
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        if df.shape[1] < 4:
            raise ValueError(
                f"Prediction CSV {path} must have at least 4 columns; got {df.shape[1]}"
            )

        col = pd.to_numeric(df.iloc[:, 3], errors="coerce")
        if col.isna().any():
            raise ValueError(
                f"Prediction CSV {path} has non-numeric values in 4th column (voicing)."
            )

        arr = (col.to_numpy(dtype=np.float64) > 0.5).astype(np.float64)
        return arr

    if suffix == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr).squeeze()

        if arr.ndim != 1:
            raise ValueError(f"Expected a 1D activation array in {path}, got shape {arr.shape}")

        if np.any(np.isnan(arr)):
            raise ValueError(f"Found NaNs in activation file: {path}")

        arr = np.clip(arr.astype(np.float64), 0.0, 1.0)
        return arr

    raise ValueError(f"Unsupported prediction format: {path}. Use .csv or .npy")


def load_note_intervals(path: Path) -> np.ndarray:
    """
    Load note intervals from a CSV file with columns: onset, offset
    Returns an array of shape (n_notes, 2).
    """
    df = pd.read_csv(path)

    column_map = {c.lower(): c for c in df.columns}
    onset_col = column_map.get("onset")
    offset_col = column_map.get("offset")

    if onset_col is None or offset_col is None:
        raise ValueError(
            f"Ground-truth file {path} must contain 'onset' and 'offset' columns."
        )

    intervals = df[[onset_col, offset_col]].to_numpy(dtype=np.float64)

    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError(f"Invalid interval format in {path}")

    valid = np.isfinite(intervals).all(axis=1) & (intervals[:, 1] > intervals[:, 0])
    intervals = intervals[valid]

    return intervals


def intervals_to_frame_mask(
    intervals: np.ndarray,
    n_frames: int,
    hop_size: float,
) -> np.ndarray:
    """
    Convert note intervals into a frame-wise voiced/unvoiced mask.

    Frame k covers:
        [k * hop_size, (k + 1) * hop_size)

    A frame is marked as voiced if it overlaps at least one note interval.
    """
    mask = np.zeros(n_frames, dtype=bool)

    if len(intervals) == 0:
        return mask

    for onset, offset in intervals:
        start_idx = max(0, int(math.floor(onset / hop_size)))
        end_idx = min(n_frames, int(math.ceil((offset - 1e-12) / hop_size)))
        if end_idx > start_idx:
            mask[start_idx:end_idx] = True

    return mask


# ============================================================
# Dataset discovery
# ============================================================

def discover_tracks(pred_dir: Path, gt_dir: Path) -> List[str]:
    """
    Discover tracks by intersecting prediction files (.csv/.npy) and .csv ground-truth files.
    Track ID is assumed to be the stem of each filename.
    """
    pred_csv_ids = {p.stem for p in pred_dir.glob("*.csv")}
    pred_npy_ids = {p.stem for p in pred_dir.glob("*.npy")}
    pred_ids = pred_csv_ids | pred_npy_ids
    gt_ids = {p.stem for p in gt_dir.glob("*.csv")}

    common_ids = sorted(pred_ids & gt_ids)

    if not common_ids:
        raise RuntimeError(
            f"No matching track IDs found between:\n"
            f"  predictions: {pred_dir}\n"
            f"  ground truth: {gt_dir}\n"
            f"Expected matching filenames such as:\n"
            f"  predictions: cantaor_001.csv\n"
            f"  ground truth: cantaor_001.csv"
        )

    missing_pred = sorted(gt_ids - pred_ids)
    missing_gt = sorted(pred_ids - gt_ids)

    if missing_pred:
        print(f"[Warning] {len(missing_pred)} ground-truth files have no matching prediction file.")
    if missing_gt:
        print(f"[Warning] {len(missing_gt)} prediction files have no matching ground-truth file.")

    return common_ids


def resolve_prediction_path(pred_dir: Path, track_id: str) -> Path:
    """
    Resolve prediction file for a track.
    Preference order: .csv, then .npy.
    """
    csv_path = pred_dir / f"{track_id}.csv"
    if csv_path.exists():
        return csv_path

    npy_path = pred_dir / f"{track_id}.npy"
    if npy_path.exists():
        return npy_path

    raise FileNotFoundError(f"No prediction file found for track '{track_id}' in {pred_dir}")


def build_tracks(pred_dir: Path, gt_dir: Path, hop_size: float) -> List[TrackData]:
    track_ids = discover_tracks(pred_dir, gt_dir)
    tracks: List[TrackData] = []

    for track_id in track_ids:
        pred_path = resolve_prediction_path(pred_dir, track_id)
        gt_path = gt_dir / f"{track_id}.csv"

        activations = load_predictions(pred_path)
        n_frames = len(activations)

        intervals = load_note_intervals(gt_path)
        gt_voiced = intervals_to_frame_mask(intervals, n_frames=n_frames, hop_size=hop_size)

        tracks.append(
            TrackData(
                track_id=track_id,
                activations=activations,
                gt_voiced=gt_voiced,
                voiced_ratio=float(gt_voiced.mean()) if n_frames > 0 else 0.0,
            )
        )

    return tracks


# ============================================================
# Metrics
# ============================================================

def confusion_counts(gt_voiced: np.ndarray, pred_voiced: np.ndarray) -> Dict[str, int]:
    gt_voiced = np.asarray(gt_voiced, dtype=bool)
    pred_voiced = np.asarray(pred_voiced, dtype=bool)

    if gt_voiced.shape != pred_voiced.shape:
        raise ValueError(f"Shape mismatch: gt={gt_voiced.shape}, pred={pred_voiced.shape}")

    tp = int(np.sum(gt_voiced & pred_voiced))
    fp = int(np.sum(~gt_voiced & pred_voiced))
    fn = int(np.sum(gt_voiced & ~pred_voiced))
    tn = int(np.sum(~gt_voiced & ~pred_voiced))

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate_voicing_with_mir_eval(
    gt_voiced: np.ndarray,
    pred_voiced: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate frame-wise voicing using mir_eval for:
      - voicing recall
      - voicing false alarm

    And manually compute:
      - Voicing Precision (Pr-V)
      - Voicing F-measure (FM-V)

    This reproduces the paper-style voicing evaluation.
    """
    gt_voiced = np.asarray(gt_voiced, dtype=bool)
    pred_voiced = np.asarray(pred_voiced, dtype=bool)

    rec_v, false_alarm = mir_eval.melody.voicing_measures(gt_voiced, pred_voiced)

    counts = confusion_counts(gt_voiced, pred_voiced)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    pr_v = safe_div(tp, tp + fp)
    fm_v = harmonic_f1(pr_v, rec_v)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pr_v": pr_v,
        "rec_v": float(rec_v),
        "fm_v": fm_v,
        "false_alarm": float(false_alarm),
        "n_frames": len(gt_voiced),
        "gt_voiced_frames": int(np.sum(gt_voiced)),
        "pred_voiced_frames": int(np.sum(pred_voiced)),
    }


def evaluate_threshold_dataset(tracks: List[TrackData], threshold: float) -> Dict[str, Any]:
    """
    Evaluate one threshold over the full dataset by concatenating all frames.
    This is a micro-averaged frame-wise evaluation.
    """
    all_gt = []
    all_pred = []

    for track in tracks:
        pred_voiced = track.activations >= threshold
        all_gt.append(track.gt_voiced.astype(bool))
        all_pred.append(pred_voiced.astype(bool))

    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    metrics = evaluate_voicing_with_mir_eval(all_gt, all_pred)

    return {
        "threshold": threshold,
        **metrics,
        "gt_voiced_ratio": safe_div(metrics["gt_voiced_frames"], metrics["n_frames"]),
        "pred_voiced_ratio": safe_div(metrics["pred_voiced_frames"], metrics["n_frames"]),
    }


def evaluate_threshold_per_track(tracks: List[TrackData], threshold: float) -> pd.DataFrame:
    rows = []

    for track in tracks:
        pred_voiced = track.activations >= threshold
        metrics = evaluate_voicing_with_mir_eval(track.gt_voiced, pred_voiced)

        rows.append({
            "track_id": track.track_id,
            "threshold": threshold,
            **metrics,
            "gt_voiced_ratio": safe_div(metrics["gt_voiced_frames"], metrics["n_frames"]),
            "pred_voiced_ratio": safe_div(metrics["pred_voiced_frames"], metrics["n_frames"]),
        })

    return pd.DataFrame(rows)


# ============================================================
# Threshold sweep
# ============================================================

def make_thresholds(start: float, stop: float, step: float) -> np.ndarray:
    n = int(round((stop - start) / step)) + 1
    values = start + step * np.arange(n)
    values = np.round(values, 10)
    values = values[(values >= 0.0) & (values <= 1.0)]
    return values


# ============================================================
# Plotting and reporting
# ============================================================

def plot_threshold_sweep(
    results_df: pd.DataFrame,
    baseline_pr: float,
    baseline_rec: float,
    baseline_fm: float,
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(results_df["threshold"], results_df["pr_v"], marker="o", label="Pr-V")
    plt.plot(results_df["threshold"], results_df["rec_v"], marker="o", label="Rec-V")
    plt.plot(results_df["threshold"], results_df["fm_v"], marker="o", label="FM-V")

    plt.axhline(baseline_pr, linestyle="--", linewidth=1, label=f"Baseline Pr-V={baseline_pr:.2f}")
    plt.axhline(baseline_rec, linestyle="--", linewidth=1, label=f"Baseline Rec-V={baseline_rec:.2f}")
    plt.axhline(baseline_fm, linestyle="--", linewidth=1, label=f"Baseline FM-V={baseline_fm:.2f}")

    best_idx = results_df["fm_v"].idxmax()
    best_row = results_df.loc[best_idx]

    plt.axvline(
        best_row["threshold"],
        linestyle=":",
        linewidth=1.5,
        label=f"Best threshold={best_row['threshold']:.2f}"
    )

    plt.xlabel("Voicing threshold")
    plt.ylabel("Metric value")
    plt.title("Cante2MIDI voicing threshold sweep")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def print_summary(
    best_row: pd.Series,
    baseline_pr: float,
    baseline_rec: float,
    baseline_fm: float,
    global_gt_voiced_ratio: float,
) -> None:
    print("\n=== Best threshold on Cante2MIDI ===")
    print(f"Threshold : {best_row['threshold']:.3f}")
    print(f"Pr-V      : {best_row['pr_v']:.4f}")
    print(f"Rec-V     : {best_row['rec_v']:.4f}")
    print(f"FM-V      : {best_row['fm_v']:.4f}")
    print(f"FA-V      : {best_row['false_alarm']:.4f}")

    print("\n=== Comparison against Kroher & Gómez baseline ===")
    print(f"Baseline Pr-V : {baseline_pr:.4f} | Delta: {best_row['pr_v'] - baseline_pr:+.4f}")
    print(f"Baseline Rec-V: {baseline_rec:.4f} | Delta: {best_row['rec_v'] - baseline_rec:+.4f}")
    print(f"Baseline FM-V : {baseline_fm:.4f} | Delta: {best_row['fm_v'] - baseline_fm:+.4f}")

    print("\n=== Ground-truth voiced-frame sanity check ===")
    print(f"Derived voiced-frame ratio: {global_gt_voiced_ratio:.4%}")
    print("Expected C2M ratio is approximately 42%.")


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_pred_dir = repo_root / "data" / "cante2midi_voicing_predictions"
    default_gt_dir = repo_root / "data" / "cante2midi_groundTruth"
    default_results_dir = repo_root / "SourceSeparation" / "results"

    parser = argparse.ArgumentParser(description="Voicing evaluation for Cante2MIDI using mir_eval")

    parser.add_argument(
        "--pred-dir",
        type=str,
        default=str(default_pred_dir),
        help="Directory containing model voicing predictions (.csv with binary 4th column, or .npy)"
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=str(default_gt_dir),
        help="Directory containing C2M ground-truth note interval CSV files"
    )
    parser.add_argument(
        "--hop-size",
        type=float,
        required=True,
        help="Frame duration in seconds, e.g. 0.01 for 10 ms"
    )

    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-stop", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)

    parser.add_argument("--baseline-pr", type=float, default=0.92)
    parser.add_argument("--baseline-rec", type=float, default=0.94)
    parser.add_argument("--baseline-fm", type=float, default=0.92)

    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(default_results_dir),
        help="Directory to save CSV and plot outputs"
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    results_dir = Path(args.results_dir)

    ensure_dir(results_dir)

    results_csv = results_dir / "voicing_threshold_results.csv"
    per_track_csv = results_dir / "per_track_best_threshold.csv"
    plot_png = results_dir / "voicing_threshold_sweep.png"

    tracks = build_tracks(pred_dir=pred_dir, gt_dir=gt_dir, hop_size=args.hop_size)

    if len(tracks) == 0:
        raise RuntimeError("No tracks were loaded.")

    thresholds = make_thresholds(
        start=args.threshold_start,
        stop=args.threshold_stop,
        step=args.threshold_step,
    )

    dataset_rows = []
    all_per_track_rows = []

    for threshold in thresholds:
        dataset_rows.append(evaluate_threshold_dataset(tracks, float(threshold)))
        all_per_track_rows.append(evaluate_threshold_per_track(tracks, float(threshold)))

    results_df = pd.DataFrame(dataset_rows).sort_values("threshold").reset_index(drop=True)
    per_track_df = pd.concat(all_per_track_rows, ignore_index=True)

    # Best threshold:
    # 1) highest FM-V
    # 2) highest Rec-V
    # 3) lowest threshold
    best_row = (
        results_df.sort_values(
            by=["fm_v", "rec_v", "threshold"],
            ascending=[False, False, True]
        )
        .reset_index(drop=True)
        .iloc[0]
    )

    best_threshold = float(best_row["threshold"])

    best_per_track_df = (
        per_track_df[per_track_df["threshold"] == best_threshold]
        .sort_values("track_id")
        .reset_index(drop=True)
    )

    results_df.to_csv(results_csv, index=False)
    best_per_track_df.to_csv(per_track_csv, index=False)

    plot_threshold_sweep(
        results_df=results_df,
        baseline_pr=args.baseline_pr,
        baseline_rec=args.baseline_rec,
        baseline_fm=args.baseline_fm,
        out_path=plot_png,
    )

    total_voiced = sum(int(np.sum(track.gt_voiced)) for track in tracks)
    total_frames = sum(len(track.gt_voiced) for track in tracks)
    global_gt_ratio = safe_div(total_voiced, total_frames)

    print_summary(
        best_row=best_row,
        baseline_pr=args.baseline_pr,
        baseline_rec=args.baseline_rec,
        baseline_fm=args.baseline_fm,
        global_gt_voiced_ratio=global_gt_ratio,
    )

    print("\n=== Saved files ===")
    print(f"Dataset sweep CSV : {results_csv}")
    print(f"Per-track CSV     : {per_track_csv}")
    print(f"Threshold plot    : {plot_png}")


if __name__ == "__main__":
    main()