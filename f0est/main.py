import argparse
import csv
from pathlib import Path

from pesto_tracker import PESTOTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Batch f0 extraction from wav files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing source wav files (or subfolders).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where f0 outputs will be stored.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["pesto"],
        default="pesto",
        help="f0 extraction method.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/voice.wav",
        help="Glob pattern to find input files inside --input-dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N files.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold used for reporting voiced frames.",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.01,
        help="Frame step size in seconds for supported methods.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate (Hz) to resample audio before f0 extraction.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU for supported methods.",
    )
    return parser.parse_args()


def build_tracker(method: str, confidence_threshold: float, step_size: float, sample_rate: int, use_gpu: bool):
    if method == "pesto":
        return PESTOTracker(
            confidence_threshold=confidence_threshold,
            step_size=step_size,
            sample_rate=sample_rate,
            use_gpu=use_gpu,
        )
    raise ValueError(f"Unsupported method: {method}")


def save_f0_csv(output_csv: Path, times, pitches, confidence):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "pitch", "confidence"])
        for t, p, c in zip(times, pitches, confidence):
            writer.writerow([float(t), float(p), float(c)])


if __name__ == "__main__":
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(args.input_dir.glob(args.pattern))
    if args.limit is not None:
        wav_files = wav_files[: args.limit]

    print(f"{len(wav_files)} wav files found.")

    tracker = build_tracker(
        method=args.method,
        confidence_threshold=args.confidence_threshold,
        step_size=args.step_size,
        sample_rate=args.sample_rate,
        use_gpu=args.use_gpu,
    )

    for wav_file in wav_files:
        pitches, times, confidence = tracker.extract_f0(wav_file)

        rel = wav_file.relative_to(args.input_dir)
        out_dir = args.output_dir / rel.parent
        out_csv = out_dir / f"{wav_file.stem}_{args.method}_f0.csv"
        save_f0_csv(out_csv, times, pitches, confidence)

        voiced = (confidence > args.confidence_threshold).sum()
        print(f"[{args.method}] {wav_file} -> {out_csv}")
        print(f"Extracted {len(pitches)} pitch frames | Voiced frames: {voiced}")