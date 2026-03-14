import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from Separator import DemucsSeparator, SpleeterSeparator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch source separation with Demucs or Spleeter."
    )

    parser.add_argument(
        "--model",
        choices=["demucs", "spleeter"],
        default="demucs",
        help="Separation backend to use.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Folder containing input audio files (used if --csv is not provided).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where separated stems will be written.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".wav"],
        help="Audio extensions to search in --input-dir (e.g., .wav .mp3).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input folder recursively.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N files.",
    )

    # Demucs-specific
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for Demucs.",
    )

    # Spleeter-specific
    parser.add_argument(
        "--stems",
        type=int,
        choices=[2, 4, 5],
        default=2,
        help="Number of stems for Spleeter.",
    )

    # Optional dataframe mode
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV file containing audio paths.",
    )
    parser.add_argument(
        "--path-column",
        type=str,
        default="wav_path",
        help="Column in CSV that contains audio file paths.",
    )

    return parser.parse_args()


def collect_files(input_dir: Path, extensions, recursive=False):
    if input_dir is None:
        return []

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    patterns = [f"*{ext if ext.startswith('.') else '.' + ext}" for ext in extensions]
    files = []
    for p in patterns:
        files.extend(input_dir.rglob(p) if recursive else input_dir.glob(p))
    return sorted(set(files))


def main():
    args = parse_args()
    t0 = time.time()

    if args.model == "demucs":
        device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
        if args.device == "auto" and device == "auto":
            device = "cpu"
        print(f"CUDA available? {torch.cuda.is_available()}")
        separator = DemucsSeparator(output_dir=str(args.output_dir), device=device)
    else:
        separator = SpleeterSeparator(output_dir=str(args.output_dir), stems=args.stems)

    # CSV mode
    if args.csv is not None:
        df = pd.read_csv(args.csv)
        if args.path_column not in df.columns:
            raise ValueError(f"Column '{args.path_column}' not found in {args.csv}")
        if args.limit is not None:
            df = df.iloc[:args.limit]
        separator.separate_from_dataframe(df, args.path_column)
        print(f"Total elapsed time: {time.time() - t0:.2f} seconds")
        return

    # Folder mode
    if args.input_dir is None:
        raise ValueError("Provide --input-dir or --csv.")

    audio_files = collect_files(args.input_dir, args.extensions, recursive=args.recursive)
    if args.limit is not None:
        audio_files = audio_files[:args.limit]

    if not audio_files:
        print("No audio files found.")
        return

    for audio_file in audio_files:
        file_t0 = time.time()
        print(f"Processing {audio_file}")
        separator.separate_file(audio_file)
        print(f"File processed in {time.time() - file_t0:.2f} seconds.")

    print(f"Total elapsed time: {time.time() - t0:.2f} seconds")


if __name__ == "__main__":

    #Example usage:
# python3 -m SourceSeparation.main --model demucs --input-dir /path/to/wavs --output-dir /path/to/stems --extensions .wav .mp3 --recursive
# python3 -m SourceSeparation.main --model spleeter --stems 2 --input-dir /path/to/wavs --output-dir /path/to/stems
    main()