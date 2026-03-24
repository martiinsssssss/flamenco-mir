#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from evaluate_onset_note_transcription import aggregate, evaluate_pair, save_csv


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all note transcription folders in noteTranscription/ "
            "against data/cante2midi_groundTruth."
        )
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=repo_root / "data" / "cante2midi_groundTruth",
        help="Directory with ground-truth .notes files.",
    )
    parser.add_argument(
        "--est-root",
        type=Path,
        default=repo_root / "noteTranscription",
        help="Root directory containing transcription subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "evaluation" / "results",
        help="Directory to save per-system and summary CSV files.",
    )
    parser.add_argument(
        "--folder-pattern",
        type=str,
        default="*_transcription",
        help="Glob pattern to select transcription folders under --est-root.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Evaluate at most N file pairs per folder (debug option).",
    )
    return parser.parse_args()


def get_estimation_folders(est_root: Path, pattern: str) -> List[Path]:
    folders = sorted([p for p in est_root.glob(pattern) if p.is_dir()])

    # Fallback: also include root if it directly contains .notes/.csv estimates
    has_est_files = any(est_root.glob("*.notes*")) or any(est_root.glob("*.csv"))
    if has_est_files:
        folders.append(est_root)

    # Deduplicate preserving order
    out: List[Path] = []
    seen = set()
    for f in folders:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def normalize_track_id(path: Path) -> str:
    """Normalize filename to a common track id (strip .notes/.csv suffix chain)."""
    name = path.name
    for suffix in (".notes.csv", ".notes", ".csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def build_pairs(ref_dir: Path, est_dir: Path) -> List[tuple[Path, Path]]:
    """
    Build (ref, est) pairs using normalized track ids.

    Supports estimate names like:
      <id>.notes.csv, <id>.notes, <id>.csv
    and reference names:
      <id>.notes or <id>.csv
    """
    ref_files = sorted(list(ref_dir.glob("*.notes")) + list(ref_dir.glob("*.csv")))
    if not ref_files:
        raise FileNotFoundError(f"No reference .notes/.csv files found in {ref_dir}")

    est_files = sorted(
        list(est_dir.glob("*.notes.csv")) +
        list(est_dir.glob("*.notes")) +
        list(est_dir.glob("*.csv"))
    )
    if not est_files:
        raise FileNotFoundError(f"No estimate .notes/.csv files found in {est_dir}")

    ref_map = {normalize_track_id(p): p for p in ref_files}
    est_map = {normalize_track_id(p): p for p in est_files}

    common_ids = sorted(set(ref_map.keys()) & set(est_map.keys()))
    if not common_ids:
        raise FileNotFoundError("No matching reference / estimate pairs found.")

    missing_ids = sorted(set(ref_map.keys()) - set(est_map.keys()))
    if missing_ids:
        print("Warning: missing estimate files for:")
        for tid in missing_ids:
            print(f"  - {ref_map[tid].name}")

    return [(ref_map[tid], est_map[tid]) for tid in common_ids]


def evaluate_folder(ref_dir: Path, est_dir: Path, limit_files: int | None = None) -> Dict[str, float]:
    pairs = build_pairs(ref_dir, est_dir)
    if limit_files is not None:
        pairs = pairs[:limit_files]

    results = [evaluate_pair(ref_path, est_path) for ref_path, est_path in pairs]
    summary = aggregate(results)

    return {
        "n_pairs": float(len(pairs)),
        **summary,
        "_results": results,
    }


def write_summary_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if not args.ref_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {args.ref_dir}")
    if not args.est_root.exists():
        raise FileNotFoundError(f"Estimation root directory not found: {args.est_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    folders = get_estimation_folders(args.est_root, args.folder_pattern)
    if not folders:
        raise FileNotFoundError(
            f"No estimation folders found in {args.est_root} with pattern '{args.folder_pattern}'"
        )

    summary_rows: List[Dict[str, object]] = []

    print(f"Found {len(folders)} estimation folder(s).")

    for est_dir in folders:
        print(f"\n=== Evaluating: {est_dir.name} ===")
        try:
            metrics = evaluate_folder(args.ref_dir, est_dir, args.limit_files)
            per_track_results = metrics.pop("_results")

            per_track_csv = args.output_dir / f"{est_dir.name}.per_track.csv"
            save_csv(per_track_results, per_track_csv)

            row = {
                "system": est_dir.name,
                "est_dir": str(est_dir),
                "n_pairs": int(metrics["n_pairs"]),
                "onset_f1": metrics["onset_f1"],
                "note_f1": metrics["note_f1"],
                "mir_eval_note_f1": metrics["mir_eval_note_f1"],
                "voicing_f1": metrics["voicing_f1"],
                "voicing_accuracy": metrics["voicing_accuracy"],
                "onset_precision": metrics["onset_precision"],
                "onset_recall": metrics["onset_recall"],
                "note_precision": metrics["note_precision"],
                "note_recall": metrics["note_recall"],
                "mir_eval_note_precision": metrics["mir_eval_note_precision"],
                "mir_eval_note_recall": metrics["mir_eval_note_recall"],
                "voicing_precision": metrics["voicing_precision"],
                "voicing_recall": metrics["voicing_recall"],
            }
            summary_rows.append(row)

            print(f"Pairs: {row['n_pairs']}")
            print(
                f"onset_f1={row['onset_f1']:.4f} | note_f1={row['note_f1']:.4f} | "
                f"mir_eval_note_f1={row['mir_eval_note_f1']:.4f} | voicing_f1={row['voicing_f1']:.4f}"
            )
            print(f"Saved per-track: {per_track_csv}")

        except Exception as e:
            print(f"Failed for {est_dir}: {e}")
            continue

    if not summary_rows:
        raise RuntimeError("No systems were successfully evaluated.")

    summary_csv = args.output_dir / "summary_all_systems.csv"
    write_summary_csv(summary_rows, summary_csv)

    print("\n=== Summary (all systems) ===")
    for row in summary_rows:
        print(
            f"[{row['system']}] n={row['n_pairs']} | "
            f"onset_f1={float(row['onset_f1']):.4f} | "
            f"note_f1={float(row['note_f1']):.4f} | "
            f"mir_eval_note_f1={float(row['mir_eval_note_f1']):.4f} | "
            f"voicing_f1={float(row['voicing_f1']):.4f}"
        )
    print(f"\nSaved summary: {summary_csv}")


if __name__ == "__main__":
    main()
