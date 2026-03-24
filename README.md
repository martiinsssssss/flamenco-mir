# flamenco-mir

Flamenco MIR workspace for:
- source separation (voice/instrumental)
- frame-level f0 extraction on separated voice tracks
- note transcription (onset detection + pitch labelling)
- note/onset evaluation utilities

This README includes setup and **all commands to run the four main entry points**:
- [SourceSeparation/main.py](SourceSeparation/main.py)
- [f0est/main.py](f0est/main.py)
- [noteTranscription/main.py](noteTranscription/main.py)
- [evaluation/main.py](evaluation/main.py)

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
- Pre-computed features (for note transcription):
  - `data/cante2midi_f0/` (f0 contours)
  - `data/cante2midi_spectrum/` (magnitude spectra)
  - `data/cante2midi_lowlevel/` (RMS and other low-level features)
- Note transcription outputs (current folders):
  - `noteTranscription/baseline_f0_transcription/`
  - `noteTranscription/baseline_f0_canteTranscription/`
  - `noteTranscription/demucs_PESTO_transcription/`
  - `noteTranscription/demucs_PESTO_canteTranscription/`
  - `noteTranscription/spleeter_PESTO_transcription/`
  - `noteTranscription/spleeter_PESTO_canteTranscription/`
- Evaluation outputs (recommended): `evaluation/results/`

### 2.1 How the data is organized (important)

The project expects all dataset parts to be placed directly under `data/` at the same directory level.

Please download **both ZIP files**:
- cante metadata ZIP
- cante audio ZIP

Then unzip both of them so that all `feature_level` folders are inside `data/` (not nested inside an extra ZIP folder), under the flamenco-mir folder.

Expected layout (simplified):

```text
data/
  cante2midiaudio/
  cante2midi_meta.xml
  cante2midi_groundTruth/
  cante2midi_f0/
  cante2midi_lowlevel/
  cante2midi_spectrum/
  ...
```

If a ZIP creates an extra nested folder (for example `data/some_zip_root/cante2midi_f0`), move the inner folders up so paths like `data/cante2midi_f0` exist exactly.

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

## 5) Run note transcription (`noteTranscription/main.py`)

Script: [noteTranscription/main.py](noteTranscription/main.py)

This script supports two transcription methods:
- `paper`: Kroher & Gómez method using pre-computed `f0 + lowlevel + spectrum` features.
- `cante`: PyCante transcription using `audio + f0` pairs.

### 5.1 Basic run (`paper` method)

```bash
python noteTranscription/main.py \
  --method paper \
  --f0-dir data/cante2midi_f0 \
  --lowlevel-dir data/cante2midi_lowlevel \
  --spectrum-dir data/cante2midi_spectrum \
  --output-dir noteTranscription/output
```

### 5.2 Process only first N tracks (`paper` method)

```bash
python noteTranscription/main.py \
  --method paper \
  --f0-dir data/cante2midi_f0 \
  --lowlevel-dir data/cante2midi_lowlevel \
  --spectrum-dir data/cante2midi_spectrum \
  --output-dir noteTranscription/output \
  --limit 10
```

### 5.3 Basic run (`cante` method)

```bash
python noteTranscription/main.py \
  --method cante \
  --audio-dir data/cante2midiaudio \
  --f0-dir f0est/f0Contours_2col/f0Contour_voiceDemucs_PESTO \
  --pycante-path ../PyCante \
  --cante-acc \
  --output-dir noteTranscription/output_cante
```

If `cante` is already installed in the active environment, you can omit `--pycante-path`.

### 5.4 Important arguments

- `--method {paper,cante}`
- `--f0-dir <dir>` (default: `data/cante2midi_f0`)
- `--audio-dir <dir>` (used by `cante`, default: `data/cante2midiaudio`)
- `--pycante-path <dir>` (optional path to PyCante source)
- `--cante-acc` (enable accompaniment-aware mode in `cante`)
- `--lowlevel-dir <dir>` (used by `paper`, default: `data/cante2midi_lowlevel`)
- `--spectrum-dir <dir>` (used by `paper`, default: `data/cante2midi_spectrum`)
- `--output-dir <dir>` (required)
- `--fs 44100` (sample rate)
- `--hop-size 128` (hop size in samples)
- `--limit N` (process first N files)
- `--delta-p-min 80.0` (pitch onset threshold in cents)
- `--gauss-sigma-s 0.0435` (Gaussian filter sigma in seconds)
- `--min-duration-s 0.05` (minimum note duration)

### 5.5 Output format

Each output CSV (`<track_id>.notes.csv`) contains:
- `onset` (seconds)
- `duration` (seconds)
- `midi` (MIDI pitch number)

Example output:
```
onset,duration,midi
0.636,0.075,56
0.711,0.058,56
0.769,0.073,56
0.842,0.070,60
```

---

## 6) Run evaluation (`evaluation/main.py`)

Script: [evaluation/main.py](evaluation/main.py)

This script evaluates all transcription folders under `noteTranscription/*_transcription`
against the ground truth in `data/cante2midi_groundTruth`.

### 6.1 Evaluate all systems

```bash
python evaluation/main.py
```

### 6.2 Quick test (first N file pairs per system)

```bash
python evaluation/main.py --limit-files 5
```

### 6.3 Custom paths

```bash
python evaluation/main.py \
  --ref-dir data/cante2midi_groundTruth \
  --est-root noteTranscription \
  --folder-pattern "*_transcription" \
  --output-dir evaluation/results
```

### 6.4 Outputs

- `evaluation/results/summary_all_systems.csv`
- `evaluation/results/<system>.per_track.csv`

---

## 7) End-to-end example

### Option A: Use pre-computed features from dataset

```bash
# Run note transcription on all cante2midi tracks
python noteTranscription/main.py \
  --f0-dir data/cante2midi_f0 \
  --lowlevel-dir data/cante2midi_lowlevel \
  --spectrum-dir data/cante2midi_spectrum \
  --output-dir noteTranscription/output
```

### Option B: Full pipeline (separation → f0 → transcription)

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

# 3) Note: You'll need to compute spectrum and RMS features 
#    separately or download them through the canto2midi dataset page.
#    Then run note transcription with custom paths:
python noteTranscription/main.py \
  --f0-dir <your_f0_output_dir> \
  --lowlevel-dir <your_rms_output_dir> \
  --spectrum-dir <your_spectrum_output_dir> \
  --output-dir noteTranscription/output
```

---

## 8) Quick troubleshooting

- If a command finds no files, verify `--input-dir`, `--extensions`, and `--pattern`.
- If GPU is unavailable, use Demucs with `--device cpu` and omit `--use-gpu` in f0 extraction.
- Ensure `ffmpeg` is available in the environment for audio tooling compatibility.
- For note transcription: ensure all three directories (f0, lowlevel, spectrum) contain matching file names (e.g., `01_Artist_Song.f0.csv`, `01_Artist_Song.lowlevel.csv`, etc.)
- For evaluation: estimate files can be `.notes.csv`, `.notes`, or `.csv`; matching is done by track id.