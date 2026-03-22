# flamenco-mir

Flamenco MIR workspace for:
- source separation (voice/instrumental)
- frame-level f0 extraction on separated voice tracks
- note/onset evaluation utilities

This README includes setup and **all commands to run the two main entry points**:
- [SourceSeparation/main.py](SourceSeparation/main.py)
- [f0est/main.py](f0est/main.py)

---

## 1) Environment setup

From the repository root:

```bash
conda env create -f environment.yml
conda activate flamencoMIR
```

If the environment already exists:

```bash
conda activate flamencoMIR
```

Optional check:

```bash
python -V
```

---

## 2) Repository paths used by the pipelines

- Input audio (dataset): `data/cante2midiaudio/`
- Source-separation outputs (default examples):
	- `SourceSeparation/demucs_output/`
	- `SourceSeparation/spleeter_output/`
- f0 outputs (recommended): `f0est/f0Contour_voice_PESTO/`

---

## 3) Run source separation (`SourceSeparation/main.py`)

Script: [SourceSeparation/main.py](SourceSeparation/main.py)

### 3.1 Demucs (recommended baseline)

```bash
python SourceSeparation/main.py \
	--model demucs \
	--input-dir data/cante2midiaudio \
	--output-dir SourceSeparation/demucs_output \
	--extensions .wav \
	--recursive
```

### 3.2 Spleeter (2 stems)

```bash
python SourceSeparation/main.py \
	--model spleeter \
	--stems 2 \
	--input-dir data/cante2midiaudio \
	--output-dir SourceSeparation/spleeter_output \
	--extensions .wav \
	--recursive
```

### 3.3 Using a CSV list of files

CSV must contain a column with audio paths (default: `wav_path`):

```bash
python SourceSeparation/main.py \
	--model demucs \
	--csv data/my_audio_list.csv \
	--path-column wav_path \
	--output-dir SourceSeparation/demucs_output
```

### 3.4 Important arguments

- `--model {demucs,spleeter}`
- `--input-dir <dir>` or `--csv <file>` (one is required)
- `--output-dir <dir>` (required)
- `--extensions .wav .mp3 ...`
- `--recursive` to search nested folders
- `--limit N` to process first N files
- Demucs only: `--device {auto,cpu,cuda}`
- Spleeter only: `--stems {2,4,5}`

### 3.5 Output layout

Per track, the separator writes a folder under `--output-dir`, typically containing vocal and accompaniment/instrumental stems.

---

## 4) Run f0 extraction (`f0est/main.py`)

Script: [f0est/main.py](f0est/main.py)

This script batch-processes WAV files and writes one CSV per matched input.

### 4.1 Typical run on separated voice tracks

```bash
python f0est/main.py \
	--input-dir SourceSeparation/demucs_output \
	--output-dir f0est/f0Contour_voice_PESTO \
	--pattern "**/voice.wav" \
	--method pesto
```

### 4.2 If your separator uses `vocals.wav`

```bash
python f0est/main.py \
	--input-dir SourceSeparation/spleeter_output \
	--output-dir f0est/f0Contour_voice_PESTO \
	--pattern "**/vocals.wav" \
	--method pesto
```

### 4.3 Important arguments

- `--input-dir <dir>` (required)
- `--output-dir <dir>` (required)
- `--method pesto`
- `--pattern "**/voice.wav"` (or `"**/vocals.wav"`)
- `--limit N`
- `--confidence-threshold 0.5`
- `--step-size 0.00290249`
- `--sample-rate 44100`
- `--use-gpu`

### 4.4 Output format

Each output CSV contains:
- `time_sec`
- `f0_hz`
- `confidence`

---

## 5) End-to-end example

Run separation first, then f0:

```bash
# 1) Separate
python SourceSeparation/main.py \
	--model demucs \
	--input-dir data/cante2midiaudio \
	--output-dir SourceSeparation/demucs_output \
	--extensions .wav \
	--recursive

# 2) Extract f0 from separated voice stems
python f0est/main.py \
	--input-dir SourceSeparation/demucs_output \
	--output-dir f0est/f0Contour_voice_PESTO \
	--pattern "**/voice.wav" \
	--method pesto
```

---

## 6) Quick troubleshooting

- If a command finds no files, verify `--input-dir`, `--extensions`, and `--pattern`.
- If GPU is unavailable, use Demucs with `--device cpu` and omit `--use-gpu` in f0 extraction.
- Ensure `ffmpeg` is available in the environment for audio tooling compatibility.