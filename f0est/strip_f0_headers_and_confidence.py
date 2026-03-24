import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert f0Contour_*.f0.csv files to 2-column CSV format "
            "(time,f0) without header, matching data/cante2midi_f0 style."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing f0Contour_* folders.",
    )
    parser.add_argument(
        "--folder-pattern",
        type=str,
        default="f0Contour_*",
        help="Glob pattern used to find source folders.",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.f0.csv",
        help="Glob pattern used to find CSV files inside each source folder.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite source files. If omitted, writes converted files to --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination root for converted files (ignored with --in-place). "
            "Default: <root-dir>/f0Contours_2col"
        ),
    )
    return parser.parse_args()


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def convert_file(src_file: Path, dst_file: Path) -> int:
    dst_file.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with src_file.open("r", newline="") as f:
        reader = csv.reader(f)
        rows_out = []

        for i, row in enumerate(reader):
            if len(row) < 2:
                continue

            if i == 0 and (not _is_float(row[0]) or not _is_float(row[1])):
                # Skip header row
                continue

            rows_out.append(f"{row[0]},{row[1]}")

    with dst_file.open("w", newline="") as f:
        if rows_out:
            f.write("\n".join(rows_out) + "\n")
            written = len(rows_out)

    return written


def main() -> None:
    args = parse_args()

    root_dir = args.root_dir.resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    source_folders = sorted(p for p in root_dir.glob(args.folder_pattern) if p.is_dir())
    if not source_folders:
        print(f"No folders found in {root_dir} with pattern: {args.folder_pattern}")
        return

    output_dir = None
    if not args.in_place:
        output_dir = (args.output_dir or (root_dir / "f0Contours_2col")).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_rows = 0

    for folder in source_folders:
        csv_files = sorted(folder.glob(args.file_pattern))
        if not csv_files:
            print(f"No files matched in {folder} using pattern: {args.file_pattern}")
            continue

        for src_file in csv_files:
            if args.in_place:
                dst_file = src_file
            else:
                dst_file = output_dir / folder.name / src_file.name

            rows = convert_file(src_file, dst_file)
            total_files += 1
            total_rows += rows
            print(f"{src_file} -> {dst_file} ({rows} rows)")

    print(f"Done. Converted {total_files} files, wrote {total_rows} rows.")


if __name__ == "__main__":
    main()
