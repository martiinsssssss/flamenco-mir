#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import mir_eval
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class NoteData:
    intervals: np.ndarray  # (N, 2) onset, offset in seconds
    midi: np.ndarray       # (N,) integer MIDI
    extra: np.ndarray | None = None

    @property
    def onsets(self) -> np.ndarray:
        return self.intervals[:, 0] if len(self.intervals) else np.array([], dtype=float)

    @property
    def offsets(self) -> np.ndarray:
        return self.intervals[:, 1] if len(self.intervals) else np.array([], dtype=float)

    @property
    def durations(self) -> np.ndarray:
        return self.offsets - self.onsets


def _empty_notedata() -> NoteData:
    return NoteData(
        intervals=np.zeros((0, 2), dtype=float),
        midi=np.zeros((0,), dtype=int),
        extra=None,
    )


def load_notes_file(path: Path) -> NoteData:
    """
    Load cante2midi .notes files or headered CSV files.

    Supported formats:
    1) cante2midi .notes (no header), one note per line:
           onset,duration,midi,extra
       Example from your ground truth:
           47.267,0.30729,64,0

    2) headered CSV:
           onset,offset,midi
       or
           onset,duration,midi

    The 4th column in .notes is preserved but ignored in evaluation.
    """
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return _empty_notedata()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    first = lines[0]

    intervals: List[List[float]] = []
    midi: List[int] = []
    extra: List[float] = []

    def add_note(onset: float, offset: float, pitch: float, extra_value: float | None = None) -> None:
        if offset <= onset:
            return
        intervals.append([float(onset), float(offset)])
        midi.append(int(round(float(pitch))))
        if extra_value is not None:
            extra.append(float(extra_value))

    # Format 1: headered CSV
    if any(tok in first.lower() for tok in ["onset", "offset", "duration", "midi"]):
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            if not {"onset", "midi"}.issubset(fieldnames):
                raise ValueError(f"{path} must contain at least onset and midi columns")

            has_offset = "offset" in fieldnames
            has_duration = "duration" in fieldnames
            if not (has_offset or has_duration):
                raise ValueError(f"{path} must contain either offset or duration column")

            for row in reader:
                onset = float(row["onset"])
                if has_offset:
                    offset = float(row["offset"])
                else:
                    offset = onset + float(row["duration"])
                pitch = float(row["midi"])
                add_note(onset, offset, pitch)
    else:
        # Format 2: cante2midi .notes without header
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                raise ValueError(f"Malformed line in {path}: {line}")
            onset = float(parts[0])
            duration = float(parts[1])
            pitch = float(parts[2])
            extra_value = float(parts[3]) if len(parts) >= 4 else None
            add_note(onset, onset + duration, pitch, extra_value)

    if not intervals:
        return _empty_notedata()

    intervals_arr = np.asarray(intervals, dtype=float)
    midi_arr = np.asarray(midi, dtype=int)
    order = np.argsort(intervals_arr[:, 0])
    intervals_arr = intervals_arr[order]
    midi_arr = midi_arr[order]
    extra_arr = np.asarray(extra, dtype=float)[order] if extra else None

    return NoteData(intervals=intervals_arr, midi=midi_arr, extra=extra_arr)


def precision_recall_f1(tp: int, n_est: int, n_ref: int) -> Tuple[float, float, float]:
    precision = tp / n_est if n_est else 0.0
    recall = tp / n_ref if n_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def onset_metrics(ref_onsets: np.ndarray, est_onsets: np.ndarray, window: float = 0.15) -> Dict[str, float]:
    f1, p, r = mir_eval.onset.f_measure(ref_onsets, est_onsets, window=window)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "n_ref": int(len(ref_onsets)),
        "n_est": int(len(est_onsets)),
    }


def notes_to_intervals(notes: np.ndarray) -> np.ndarray:
    """
    Convert notes from [onset, duration, ...] rows to [start, end] rows.
    """
    if notes.size == 0:
        return np.zeros((0, 2), dtype=float)

    intervals = []
    for n in notes:
        onset = float(n[0])
        duration = float(n[1])
        intervals.append([onset, onset + duration])
    return np.asarray(intervals, dtype=float)


def intervals_to_voicing(intervals: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    Convert note intervals to a binary voicing array over a time grid.
    """
    voicing = np.zeros_like(t_grid, dtype=int)

    for start, end in intervals:
        voicing[(t_grid >= start) & (t_grid < end)] = 1

    return voicing


def evaluate_voicing(gt_notes: np.ndarray, est_notes: np.ndarray, hop: float = 0.01) -> Dict[str, float]:
    """
    Evaluate frame-level voicing metrics from note arrays [onset, duration, ...].
    """
    gt_intervals = notes_to_intervals(gt_notes)
    est_intervals = notes_to_intervals(est_notes)

    if gt_intervals.size == 0 and est_intervals.size == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "accuracy": 1.0,
        }

    starts = []
    ends = []
    if gt_intervals.size:
        starts.append(float(gt_intervals[:, 0].min()))
        ends.append(float(gt_intervals[:, 1].max()))
    if est_intervals.size:
        starts.append(float(est_intervals[:, 0].min()))
        ends.append(float(est_intervals[:, 1].max()))

    t_min = min(starts)
    t_max = max(ends)
    if t_max <= t_min:
        t_max = t_min + hop

    t_grid = np.arange(t_min, t_max, hop)
    if t_grid.size == 0:
        t_grid = np.array([t_min], dtype=float)

    gt_voicing = intervals_to_voicing(gt_intervals, t_grid)
    est_voicing = intervals_to_voicing(est_intervals, t_grid)

    precision = precision_score(gt_voicing, est_voicing, zero_division=0)
    recall = recall_score(gt_voicing, est_voicing, zero_division=0)
    f1 = f1_score(gt_voicing, est_voicing, zero_division=0)
    acc = accuracy_score(gt_voicing, est_voicing)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }


def build_note_cost_matrix(
    ref: NoteData,
    est: NoteData,
    onset_tolerance: float = 0.15,
    duration_tolerance_ratio: float = 0.30,
    pitch_tolerance_semitones: int = 0,
) -> np.ndarray:
    n_ref = len(ref.midi)
    n_est = len(est.midi)
    if n_ref == 0 or n_est == 0:
        return np.zeros((n_ref, n_est), dtype=float)

    big = 1e9
    cost = np.full((n_ref, n_est), big, dtype=float)

    for i in range(n_ref):
        ref_dur = ref.durations[i]
        for j in range(n_est):
            onset_ok = abs(ref.onsets[i] - est.onsets[j]) <= onset_tolerance
            pitch_ok = abs(int(ref.midi[i]) - int(est.midi[j])) <= pitch_tolerance_semitones
            dur_ok = abs(ref_dur - est.durations[j]) <= duration_tolerance_ratio * ref_dur
            if onset_ok and pitch_ok and dur_ok:
                cost[i, j] = abs(ref.onsets[i] - est.onsets[j])

    return cost


def note_metrics_paper_style(
    ref: NoteData,
    est: NoteData,
    onset_tolerance: float = 0.15,
    duration_tolerance_ratio: float = 0.30,
    semitone_shifts: Tuple[int, ...] = (-1, 0, 1),
) -> Dict[str, float]:
    best: Dict[str, float] | None = None

    for shift in semitone_shifts:
        shifted = NoteData(intervals=est.intervals.copy(), midi=est.midi + shift, extra=est.extra)
        cost = build_note_cost_matrix(
            ref,
            shifted,
            onset_tolerance=onset_tolerance,
            duration_tolerance_ratio=duration_tolerance_ratio,
            pitch_tolerance_semitones=0,
        )

        if cost.size == 0:
            tp = 0
        else:
            row_ind, col_ind = linear_sum_assignment(cost)
            valid = cost[row_ind, col_ind] < 1e8
            tp = int(np.sum(valid))

        p, r, f1 = precision_recall_f1(tp, len(shifted.midi), len(ref.midi))
        result = {
            "shift": int(shift),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "tp": int(tp),
            "n_ref": int(len(ref.midi)),
            "n_est": int(len(shifted.midi)),
        }
        if best is None or result["f1"] > best["f1"]:
            best = result

    assert best is not None
    return best


def note_metrics_mir_eval(ref: NoteData, est: NoteData, onset_tolerance: float = 0.15) -> Dict[str, float]:
    ref_hz = mir_eval.util.midi_to_hz(ref.midi.astype(float))
    est_hz = mir_eval.util.midi_to_hz(est.midi.astype(float))
    p, r, f1, overlap = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals=ref.intervals,
        ref_pitches=ref_hz,
        est_intervals=est.intervals,
        est_pitches=est_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50.0,
        offset_ratio=0.30,
        offset_min_tolerance=0.0,
        strict=False,
    )


    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "avg_overlap_ratio": float(overlap),
    }



def evaluate_pair(ref_path: Path, est_path: Path) -> Dict[str, object]:
    ref = load_notes_file(ref_path)
    est = load_notes_file(est_path)

    onset = onset_metrics(ref.onsets, est.onsets, window=0.15)
    note = note_metrics_paper_style(ref, est, onset_tolerance=0.15, duration_tolerance_ratio=0.30)
    note_mir = note_metrics_mir_eval(ref, est, onset_tolerance=0.15)
    gt_notes = np.column_stack((ref.onsets, ref.durations)) if len(ref.intervals) else np.zeros((0, 2), dtype=float)
    est_notes = np.column_stack((est.onsets, est.durations)) if len(est.intervals) else np.zeros((0, 2), dtype=float)
    voicing = evaluate_voicing(gt_notes, est_notes, hop=0.01)

    return {
        "track": ref_path.stem,
        "ref_file": str(ref_path),
        "est_file": str(est_path),
        "n_ref_notes": len(ref.midi),
        "n_est_notes": len(est.midi),
        "onset_precision": onset["precision"],
        "onset_recall": onset["recall"],
        "onset_f1": onset["f1"],
        "note_precision": note["precision"],
        "note_recall": note["recall"],
        "note_f1": note["f1"],
        "best_shift_semitones": note["shift"],
        "mir_eval_note_precision": note_mir["precision"],
        "mir_eval_note_recall": note_mir["recall"],
        "mir_eval_note_f1": note_mir["f1"],
        "mir_eval_overlap": note_mir["avg_overlap_ratio"],
        "voicing_precision": voicing["precision"],
        "voicing_recall": voicing["recall"],
        "voicing_f1": voicing["f1"],
        "voicing_accuracy": voicing["accuracy"],
    }


def find_pairs(ref_dir: Path, est_dir: Path, ref_suffixes: Tuple[str, ...] = (".notes", ".csv")) -> List[Tuple[Path, Path]]:
    ref_files: List[Path] = []
    for suffix in ref_suffixes:
        ref_files.extend(sorted(ref_dir.glob(f"*{suffix}")))
    if not ref_files:
        raise FileNotFoundError(f"No reference files found in {ref_dir}")

    pairs: List[Tuple[Path, Path]] = []
    missing: List[str] = []
    for ref_path in ref_files:
        candidates = [est_dir / ref_path.name, est_dir / f"{ref_path.stem}.notes", est_dir / f"{ref_path.stem}.csv"]
        est_path = next((c for c in candidates if c.exists()), None)
        if est_path is None:
            missing.append(ref_path.name)
        else:
            pairs.append((ref_path, est_path))

    if missing:
        print("Warning: missing estimate files for:")
        for name in missing:
            print(f"  - {name}")

    if not pairs:
        raise FileNotFoundError("No matching reference / estimate pairs found.")
    return pairs


def aggregate(results: List[Dict[str, object]]) -> Dict[str, float]:
    metric_keys = [
        "onset_precision", "onset_recall", "onset_f1",
        "note_precision", "note_recall", "note_f1",
        "mir_eval_note_precision", "mir_eval_note_recall", "mir_eval_note_f1",
        "voicing_precision", "voicing_recall", "voicing_f1", "voicing_accuracy",
    ]
    out: Dict[str, float] = {}
    for key in metric_keys:
        vals = [float(r[key]) for r in results]
        out[key] = float(np.mean(vals)) if vals else 0.0
    return out


def save_csv(results: List[Dict[str, object]], out_path: Path) -> None:
    if not results:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cante2midi note annotations / estimates.")
    parser.add_argument("--ref", type=Path, help="Reference .notes or .csv file")
    parser.add_argument("--est", type=Path, help="Estimate .notes or .csv file")
    parser.add_argument("--ref_dir", type=Path, help="Directory with reference files")
    parser.add_argument("--est_dir", type=Path, help="Directory with estimate files")
    parser.add_argument("--output_csv", type=Path, default=None, help="Optional CSV for per-track results")
    args = parser.parse_args()

    single_mode = args.ref is not None and args.est is not None
    dir_mode = args.ref_dir is not None and args.est_dir is not None

    if not (single_mode or dir_mode):
        raise SystemExit("Use either --ref and --est, or --ref_dir and --est_dir")

    if single_mode:
        result = evaluate_pair(args.ref, args.est)
        for key, value in result.items():
            print(f"{key}: {value}")
        return

    pairs = find_pairs(args.ref_dir, args.est_dir)
    results = []
    for ref_path, est_path in pairs:
        result = evaluate_pair(ref_path, est_path)
        results.append(result)
        print(
            f"[{result['track']}] onset_f1={result['onset_f1']:.4f} | "
            f"note_f1={result['note_f1']:.4f} | shift={result['best_shift_semitones']:+d}"
        )

    summary = aggregate(results)
    print("\n=== Macro average ===")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    if args.output_csv is not None:
        save_csv(results, args.output_csv)
        print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
