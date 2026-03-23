#!/usr/bin/env python3
"""
Note Transcription - Batch pipeline
Computes note transcription for all f0 contours in a folder.

Matches files by base name:
  - data/cante2midi_f0/<ID>.f0.csv
      Format: time [s], f0 [Hz]
      Hop size: 128 samples @ 44.1 kHz
  
  - data/cante2midi_lowlevel/<ID>.lowlevel.csv
      Format: time [s], spectral flux, spectral roll-off, spectral complexity,
               spectral flatness, spectral centroid, RMS, zero-crossing rate
      Hop size: 512 samples @ 44.1 kHz
  
  - data/cante2midi_spectrum/<ID>.spectrum.csv
      Format: time [s], magnitude bin 1, ..., magnitude bin 512
      Hop size: 512 samples @ 44.1 kHz

Then calls transcribe_notes() and saves output as <ID>.notes.csv

Usage:
    python noteTranscription/main.py \
        --f0-dir data/cante2midi_f0 \
        --lowlevel-dir data/cante2midi_lowlevel \
        --spectrum-dir data/cante2midi_spectrum \
        --output-dir noteTranscription/output \
        --limit 10
"""

import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np

from note_transcription import check_and_align_features, transcribe_notes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch note transcription from feature files."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["paper", "cante"],
        default="paper",
        help="Transcription method: 'paper' (local implementation) or 'cante' (PyCante).",
    )
    parser.add_argument(
        "--f0-dir",
        type=Path,
        default=Path("data/cante2midi_f0"),
        help="Directory containing f0 CSV files (*.f0.csv).",
    )
    parser.add_argument(
        "--lowlevel-dir",
        type=Path,
        default=Path("data/cante2midi_lowlevel"),
        help="Directory containing lowlevel feature CSV files (*.lowlevel.csv).",
    )
    parser.add_argument(
        "--spectrum-dir",
        type=Path,
        default=Path("data/cante2midi_spectrum"),
        help="Directory containing spectrum CSV files (*.spectrum.csv).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/cante2midiaudio"),
        help="Directory containing audio WAV files (*.wav). Required for --method cante.",
    )
    parser.add_argument(
        "--pycante-path",
        type=Path,
        default="/home/ibroto/Documents/PyCante",
        help="Optional path added to sys.path before importing cante (e.g., /path/to/PyCante).",
    )
    parser.add_argument(
        "--cante-acc",
        action="store_true",
        help="Enable accompaniment-aware mode in cante.transcribe(..., acc=True).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for note transcription files (*.notes.csv).",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=44100,
        help="Sample rate in Hz.",
    )
    parser.add_argument(
        "--hop-size",
        type=int,
        default=128,
        help="Target hop size in samples used by transcription.",
    )
    parser.add_argument(
        "--f0-hop",
        type=int,
        default=128,
        help="Hop size (samples) used by f0 files.",
    )
    parser.add_argument(
        "--rms-hop",
        type=int,
        default=512,
        help="Hop size (samples) used by lowlevel/RMS files.",
    )
    parser.add_argument(
        "--spectrum-hop",
        type=int,
        default=512,
        help="Hop size (samples) used by spectrum files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N files.",
    )
    parser.add_argument(
        "--delta-p-min",
        type=float,
        default=80.0,
        help="Pitch onset detection threshold (cents).",
    )
    parser.add_argument(
        "--gauss-sigma-s",
        type=float,
        default=0.0435,
        help="Gaussian derivative sigma (seconds).",
    )
    parser.add_argument(
        "--gauss-threshold",
        type=float,
        default=4.0,
        help="Gaussian derivative threshold.",
    )
    parser.add_argument(
        "--volume-threshold-db",
        type=float,
        default=-10.0,
        help="Volume dip threshold (dB).",
    )
    parser.add_argument(
        "--pitch-z-threshold",
        type=float,
        default=-2.0,
        help="Pitch z-score threshold.",
    )
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=0.05,
        help="Minimum note duration (seconds).",
    )
    parser.add_argument(
        "--pitch-range-semitones",
        type=int,
        default=8,
        help="Pitch range for post-processing (semitones).",
    )

    return parser.parse_args()


def load_f0(f0_path: Path) -> np.ndarray:
    """
    Load f0 from CSV.
    Supported formats:
      1) No header: time_sec,f0_hz
      2) Headered CSV from f0est/main.py: time_sec,f0_hz,confidence

    Returns: f0_hz array
    """
    first_line = f0_path.open("r", encoding="utf-8").readline().strip().lower()
    has_header = any(ch.isalpha() for ch in first_line)

    if has_header:
        data = np.genfromtxt(f0_path, delimiter=",", names=True, dtype=float)

        # Prefer explicit column name if present
        if "f0_hz" in (data.dtype.names or ()):  # f0est format
            return np.asarray(data["f0_hz"], dtype=float)

        # Fallback: second column
        if data.dtype.names and len(data.dtype.names) >= 2:
            return np.asarray(data[data.dtype.names[1]], dtype=float)

        raise ValueError(f"Could not parse headered f0 CSV: {f0_path}")

    data = np.loadtxt(f0_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in f0 CSV: {f0_path}")
    # Return second column (f0 in Hz)
    return data[:, 1].astype(float)


def load_lowlevel(lowlevel_path: Path, fs: int = 44100) -> np.ndarray:
    """
    Load lowlevel features from CSV and extract RMS.
    
    Expected format (cante2midi lowlevel.csv):
    Columns: time [s], spectral flux, spectral roll-off, spectral complexity,
             spectral flatness, spectral centroid, RMS, zero-crossing rate
    
    Returns: rms array (column 6, 0-indexed)
    """
    data = np.loadtxt(lowlevel_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # Column 6 (0-indexed) is RMS according to cante2midi format
    rms = data[:, 6].astype(float)
    return rms


def load_spectrum(spectrum_path: Path) -> np.ndarray:
    """
    Load spectrum from CSV.
    Expected format: tab/space-separated magnitude spectrum frames
    First column is time, remaining columns are spectrum bins.
    Returns: spectrum array of shape (n_frames, n_bins)
    """
    data = np.loadtxt(spectrum_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # Skip first column (time) and return spectrum
    spectrum = data[:, 1:].astype(float)
    return spectrum


def find_matching_files(f0_dir: Path, lowlevel_dir: Path, spectrum_dir: Path) -> list:
    """
    Find all matching triplets (f0, lowlevel, spectrum) by base name.
    Returns list of (base_name, f0_path, lowlevel_path, spectrum_path) tuples.
    """
    f0_files = {p.stem.replace(".f0", ""): p for p in f0_dir.glob("*.f0.csv")}
    lowlevel_files = {p.stem.replace(".lowlevel", ""): p for p in lowlevel_dir.glob("*.lowlevel.csv")}
    spectrum_files = {p.stem.replace(".spectrum", ""): p for p in spectrum_dir.glob("*.spectrum.csv")}

    # Find common IDs
    common_ids = sorted(set(f0_files.keys()) & set(lowlevel_files.keys()) & set(spectrum_files.keys()))

    if not common_ids:
        raise FileNotFoundError(
            f"No matching file triplets found in:\n"
            f"  f0: {f0_dir}\n"
            f"  lowlevel: {lowlevel_dir}\n"
            f"  spectrum: {spectrum_dir}\n"
            f"Expected matching filenames like:\n"
            f"  01_Artist_Song.f0.csv\n"
            f"  01_Artist_Song.lowlevel.csv\n"
            f"  01_Artist_Song.spectrum.csv"
        )

    missing_f0 = len(lowlevel_files) + len(spectrum_files) - 2 * len(common_ids)
    if missing_f0 > 0:
        print(f"[Warning] Some files don't have matching triplets (missing {missing_f0} pairs).")

    return [(id, f0_files[id], lowlevel_files[id], spectrum_files[id]) for id in common_ids]


def find_matching_audio_f0_files(f0_dir: Path, audio_dir: Path) -> list:
    """
    Find all matching pairs (audio, f0) by base name.
    Returns list of (base_name, audio_path, f0_path) tuples.
    """
    f0_files = {p.stem.replace(".f0", ""): p for p in f0_dir.glob("*.f0.csv")}
    audio_files = {p.stem: p for p in audio_dir.glob("*.wav")}

    common_ids = sorted(set(f0_files.keys()) & set(audio_files.keys()))

    if not common_ids:
        raise FileNotFoundError(
            f"No matching audio-f0 pairs found in:\n"
            f"  audio: {audio_dir}\n"
            f"  f0: {f0_dir}\n"
            f"Expected matching filenames like:\n"
            f"  01_Artist_Song.wav\n"
            f"  01_Artist_Song.f0.csv"
        )

    missing_pairs = len(f0_files) + len(audio_files) - 2 * len(common_ids)
    if missing_pairs > 0:
        print(f"[Warning] Some files don't have matching audio-f0 pairs (missing {missing_pairs} pairs).")

    return [(id, audio_files[id], f0_files[id]) for id in common_ids]


def load_cante_module(pycante_path):
    if pycante_path is not None:
        pycante_path = pycante_path.resolve()
        if not pycante_path.exists():
            raise FileNotFoundError(f"PyCante path does not exist: {pycante_path}")
        if str(pycante_path) not in sys.path:
            sys.path.insert(0, str(pycante_path))

    try:
        return importlib.import_module("cante")
    except ImportError as e:
        raise ImportError(
            "Could not import 'cante'. Install PyCante or pass --pycante-path to its source folder."
        ) from e


def save_notes_csv(output_path: Path, notes: list) -> None:
    """
    Save note transcription to CSV.
    Format: onset, duration, midi
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["onset", "duration", "midi"])
        for note in notes:
            writer.writerow([float(note.onset), float(note.duration), int(note.midi)])


def main():
    args = parse_args()

    if not args.f0_dir.exists():
        raise FileNotFoundError(f"f0 directory not found: {args.f0_dir}")

    if args.method == "paper":
        if not args.lowlevel_dir.exists():
            raise FileNotFoundError(f"lowlevel directory not found: {args.lowlevel_dir}")
        if not args.spectrum_dir.exists():
            raise FileNotFoundError(f"spectrum directory not found: {args.spectrum_dir}")
    elif args.method == "cante":
        if not args.audio_dir.exists():
            raise FileNotFoundError(f"audio directory not found: {args.audio_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "paper":
        # Find all matching file triplets
        triplets = find_matching_files(args.f0_dir, args.lowlevel_dir, args.spectrum_dir)

        if args.limit is not None:
            triplets = triplets[: args.limit]

        print(f"{len(triplets)} file triplets found.")

        for idx, (base_id, f0_path, lowlevel_path, spectrum_path) in enumerate(triplets, 1):
            try:
                print(f"[{idx}/{len(triplets)}] Processing {base_id}...")

                # Load features
                f0_hz = load_f0(f0_path)
                rms = load_lowlevel(lowlevel_path, fs=args.fs)
                spectrum = load_spectrum(spectrum_path)

                # Align features to the same time grid (important: streams use different hop sizes)
                f0_hz, rms, spectrum = check_and_align_features(
                    f0_hz=f0_hz,
                    rms=rms,
                    spectrum=spectrum,
                    f0_hop=args.f0_hop,
                    rms_hop=args.rms_hop,
                    spectrum_hop=args.spectrum_hop,
                    target_hop=args.hop_size,
                    fs=args.fs,
                    verbose=False,
                )

                n = len(f0_hz)

                if n < 10:
                    print(f"  ⚠️  Skipping {base_id}: too few frames ({n})")
                    continue

                # Transcribe notes
                notes = transcribe_notes(
                    f0_hz=f0_hz,
                    rms=rms,
                    spectrum=spectrum,
                    fs=args.fs,
                    hop_size=args.hop_size,
                    delta_p_min=args.delta_p_min,
                    gauss_sigma_s=args.gauss_sigma_s,
                    gauss_threshold=args.gauss_threshold,
                    volume_threshold_db=args.volume_threshold_db,
                    pitch_z_threshold=args.pitch_z_threshold,
                    min_duration_s=args.min_duration_s,
                    pitch_range_semitones=args.pitch_range_semitones,
                )

                # Save
                output_path = args.output_dir / f"{base_id}.notes.csv"
                save_notes_csv(output_path, notes)

                print(f"  ✓ {len(notes)} notes transcribed -> {output_path}")

            except Exception as e:
                print(f"  ✗ Error processing {base_id}: {e}")
                continue

    elif args.method == "cante":
        cante = load_cante_module(args.pycante_path)
        pairs = find_matching_audio_f0_files(args.f0_dir, args.audio_dir)

        if args.limit is not None:
            pairs = pairs[: args.limit]

        print(f"{len(pairs)} audio-f0 pairs found.")

        for idx, (base_id, audio_path, f0_path) in enumerate(pairs, 1):
            try:
                print(f"[{idx}/{len(pairs)}] Processing {base_id}...")
                output_path = args.output_dir / f"{base_id}.notes.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                cante.transcribe(
                    str(audio_path),
                    acc=args.cante_acc,
                    f0_file=str(f0_path),
                    recursive=False,
                    output_filename=str(output_path),
                )

                print(f"  ✓ notes transcribed -> {output_path}")

            except Exception as e:
                print(f"  ✗ Error processing {base_id}: {e}")
                continue

    print(f"\nNote transcription complete ({args.method}). Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
