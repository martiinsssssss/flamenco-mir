"""
Note Transcription - Section II.B
Kroher & Gomez, "Automatic Transcription of Flamenco Singing from Polyphonic Music Recordings"

Implements:
  II.B.1 - Segmentation (4 onset detectors: local maxima, Gaussian derivative, volume dip, pitch dip)
  II.B.2 - Pitch labelling (tuning estimation, global chroma, local histogram, combined probability)
  II.B.3 - Note post-processing (pitch range clipping, octave correction, min duration filtering)

Inputs assumed available:
  - f0[n]        : predominant/vocal pitch contour in Hz (0 = unvoiced)
  - rms[n]       : frame-wise RMS of the signal
  - spectrum[n,k]: magnitude spectrum frames (for chroma computation)
  - fs           : sample rate (default 44100)
  - hop_size     : hop size in samples (default 128)
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelmax
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Note:
    onset: float      # seconds
    duration: float   # seconds
    midi: int         # MIDI pitch number


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

F_REF = 440.0  # Hz


def hz_to_cents(f0_hz: np.ndarray, f_ref: float = F_REF) -> np.ndarray:
    """Convert Hz pitch contour to cents relative to f_ref. Unvoiced (0 Hz) -> 0."""
    out = np.zeros_like(f0_hz, dtype=float)
    voiced = f0_hz > 0
    out[voiced] = 1200.0 * np.log2(f0_hz[voiced] / f_ref)
    return out


def voiced_segments(f0_hz: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start, end) frame indices for continuously voiced runs."""
    voiced = (f0_hz > 0).astype(int)
    diff = np.diff(np.concatenate(([0], voiced, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))



def check_and_align_features(
    f0_hz: np.ndarray,        # from your melody extractor
    rms: np.ndarray,          # from your RMS extractor  
    spectrum: np.ndarray,     # (n_frames, n_bins) from your STFT
    f0_hop: int,
    rms_hop: int,
    spectrum_hop: int,
    target_hop: int = 128,
    fs: int = 44100,
    verbose: bool = True,
) -> tuple:
    """
    Resamples all features to target_hop and validates consistency.
    Returns (f0, rms, spectrum) all on the same time grid.
    """

    def n_frames_to_seconds(n, hop):
        return n * hop / fs

    def resample_1d(arr, src_hop, tgt_hop):
        """Resample a 1D feature array from src_hop to tgt_hop grid."""
        if src_hop == tgt_hop:
            return arr
        src_times = np.arange(len(arr)) * src_hop / fs
        tgt_times = np.arange(int(src_times[-1] * fs / tgt_hop) + 1) * tgt_hop / fs
        tgt_times = tgt_times[tgt_times <= src_times[-1]]
        f = interp1d(src_times, arr, kind='linear', bounds_error=False,
                     fill_value=(arr[0], arr[-1]))
        return f(tgt_times)

    def resample_2d(arr, src_hop, tgt_hop):
        """Resample a 2D (n_frames, n_bins) array along the time axis."""
        if src_hop == tgt_hop:
            return arr
        result = np.stack([
            resample_1d(arr[:, k], src_hop, tgt_hop)
            for k in range(arr.shape[1])
        ], axis=1)
        return result

    # resample everything to target grid
    f0_r      = resample_1d(f0_hz,   f0_hop,       target_hop)
    rms_r     = resample_1d(rms,     rms_hop,      target_hop)
    spec_r    = resample_2d(spectrum, spectrum_hop, target_hop)

    # after resampling, lengths should match - trim to shortest
    n = min(len(f0_r), len(rms_r), len(spec_r))
    f0_r, rms_r, spec_r = f0_r[:n], rms_r[:n], spec_r[:n]

    if verbose:
        dur_f0   = n_frames_to_seconds(len(f0_hz),        f0_hop)
        dur_rms  = n_frames_to_seconds(len(rms),          rms_hop)
        dur_spec = n_frames_to_seconds(len(spectrum),     spectrum_hop)
        print(f"Input durations:  f0={dur_f0:.3f}s  rms={dur_rms:.3f}s  spec={dur_spec:.3f}s")
        print(f"After alignment:  all {n} frames = {n * target_hop / fs:.3f}s @ hop={target_hop}")
        max_drift = abs(dur_f0 - dur_rms)
        if max_drift > 0.1:
            print(f"WARNING: duration mismatch of {max_drift:.3f}s — check your extractors")

    return f0_r, rms_r, spec_r



# ---------------------------------------------------------------------------
# II.B.1  Segmentation
# ---------------------------------------------------------------------------

def detect_onsets_local_maxima(
    cents: np.ndarray,
    delta_p_min: float = 80.0,   # cents threshold between adjacent maxima
    max_time_gap: float = 0.25,  # seconds; exclude pairs further apart
    fs: int = 44100,
    hop_size: int = 128,
) -> np.ndarray:
    """
    Interval onset detection via upper-envelope local maxima (Sec. II.B.1, eq. after Eq. 7).

    The upper envelope is approximated by finding local maxima of the cent contour.
    If two adjacent maxima differ by >= delta_p_min cents AND their time distance
    is <= max_time_gap s, a note onset is placed halfway between them.
    """
    hop_s = hop_size / fs
    max_frame_gap = int(max_time_gap / hop_s)

    # local maxima of cent contour
    (maxima_idx,) = argrelmax(cents, order=3)
    if len(maxima_idx) < 2:
        return np.array([], dtype=int)

    onsets = []
    for i in range(len(maxima_idx) - 1):
        i1, i2 = maxima_idx[i], maxima_idx[i + 1]
        if (i2 - i1) > max_frame_gap:
            continue
        if abs(cents[i2] - cents[i1]) >= delta_p_min:
            onsets.append((i1 + i2) // 2)

    return np.array(onsets, dtype=int)


def detect_onsets_gaussian_derivative(
    cents: np.ndarray,
    sigma_s: float = 0.0435,   # 43.5 ms
    threshold: float = 4.0,
    fs: int = 44100,
    hop_size: int = 128,
) -> np.ndarray:
    """
    Interval onset detection via first-derivative Gaussian filter (Eq. 8-9).

    Detects long-term pitch changes while suppressing fast vibrato fluctuations.
    sigma_s = 43.5 ms => effective length ~300 ms covers one slow vibrato period at 4 Hz.
    """
    hop_s = hop_size / fs
    sigma_frames = sigma_s / hop_s

    N = int(6 * sigma_frames) | 1          # odd length, ~6-sigma window
    n = np.arange(-(N // 2), N // 2 + 1, dtype=float)
    h = -(n / sigma_frames**2) * np.exp(-(n**2) / (2 * sigma_frames**2))

    filtered = np.convolve(cents, h, mode="same")
    abs_filtered = np.abs(filtered)

    # local maxima above threshold
    (peaks,) = argrelmax(abs_filtered, order=3)
    onsets = peaks[abs_filtered[peaks] >= threshold]
    return onsets.astype(int)


def detect_onsets_volume_dip(
    rms: np.ndarray,
    threshold_db: float = -10.0,
    smoothing_frames: int = 100,   # ±50 frames ~ 150 ms
) -> np.ndarray:
    """
    Steady-pitch onset detection via local RMS dip (Eq. 10).

    r_LOC[n] = 20*log10( rms[n] / mean(rms[n-50:n+50]) )
    Onset at local minima where r_LOC < threshold_db.
    """
    # padded local mean
    local_mean = uniform_filter1d(rms, size=smoothing_frames, mode="nearest")
    local_mean = np.maximum(local_mean, 1e-10)

    r_loc = 20.0 * np.log10(np.maximum(rms, 1e-10) / local_mean)

    # local minima below threshold
    (minima_idx,) = argrelmax(-r_loc, order=3)
    onsets = minima_idx[r_loc[minima_idx] < threshold_db]
    return onsets.astype(int)


def detect_onsets_pitch_dip(
    cents: np.ndarray,
    z_threshold: float = -2.0,   # below -sqrt(2) ≈ -1.414, paper sets -2.0
) -> np.ndarray:
    """
    Steady-pitch onset detection via pitch contour dip (z-score outlier, Eq. 11-13).

    Computes z-score of the cent contour over the whole segment;
    onset candidates are local minima where z[n] < z_threshold.
    """
    if cents.std() < 1e-6:
        return np.array([], dtype=int)

    mu = cents.mean()
    sigma = cents.std()
    z = (cents - mu) / sigma

    (minima_idx,) = argrelmax(-z, order=3)
    onsets = minima_idx[z[minima_idx] < z_threshold]
    return onsets.astype(int)


def merge_and_deduplicate(
    *onset_arrays: np.ndarray,
    min_gap_frames: int = 5,
) -> np.ndarray:
    """Merge onset arrays, sort, and remove duplicates closer than min_gap_frames."""
    if not onset_arrays:
        return np.array([], dtype=int)
    list_lo_concatenate= [a for a in onset_arrays if len(a) > 0]
    all_onsets = np.concatenate(list_lo_concatenate) if len(list_lo_concatenate)>0 else []
    if len(all_onsets) == 0:
        return np.array([], dtype=int)
    all_onsets = np.sort(np.unique(all_onsets))
    # remove onsets too close together
    keep = [all_onsets[0]]
    for o in all_onsets[1:]:
        if o - keep[-1] >= min_gap_frames:
            keep.append(o)
    return np.array(keep, dtype=int)


def segment_contour(
    f0_hz: np.ndarray,
    rms: np.ndarray,
    fs: int = 44100,
    hop_size: int = 128,
    delta_p_min: float = 80.0,
    gauss_sigma_s: float = 0.0435,
    gauss_threshold: float = 4.0,
    volume_threshold_db: float = -10.0,
    pitch_z_threshold: float = -2.0,
    min_gap_s: float = 0.05,
) -> List[Tuple[int, int]]:
    """
    Full segmentation pipeline (Sec. II.B.1).

    Returns list of (onset_frame, offset_frame) pairs for each note event.
    """
    hop_s = hop_size / fs
    min_gap_frames = max(1, int(min_gap_s / hop_s))

    voiced_segs = voiced_segments(f0_hz)
    note_intervals: List[Tuple[int, int]] = []

    for seg_start, seg_end in voiced_segs:
        seg_cents = hz_to_cents(f0_hz[seg_start:seg_end])
        seg_rms = rms[seg_start:seg_end]

        # --- run all four detectors ---
        o1 = detect_onsets_local_maxima(seg_cents, delta_p_min, fs=fs, hop_size=hop_size)
        o2 = detect_onsets_gaussian_derivative(seg_cents, gauss_sigma_s, gauss_threshold, fs, hop_size)
        o3 = detect_onsets_volume_dip(seg_rms, volume_threshold_db)
        o4 = detect_onsets_pitch_dip(seg_cents, pitch_z_threshold)

        internal_onsets = merge_and_deduplicate(o1, o2, o3, o4, min_gap_frames=min_gap_frames)

        # build note boundaries within this voiced segment
        boundaries = np.concatenate(([0], internal_onsets, [len(seg_cents)]))
        for i in range(len(boundaries) - 1):
            s = int(boundaries[i]) + seg_start
            e = int(boundaries[i + 1]) + seg_start
            if e > s:
                note_intervals.append((s, e))

    return note_intervals


# ---------------------------------------------------------------------------
# II.B.2  Pitch labelling
# ---------------------------------------------------------------------------

def estimate_tuning_deviation(
    f0_hz: np.ndarray,
    bins_per_semitone: int = 10,
) -> float:
    """
    Estimate global tuning deviation in cents using circular statistics (Eq. 14, ref [36]).

    Returns delta_t in cents.
    """
    voiced = f0_hz[f0_hz > 0]
    if len(voiced) == 0:
        return 0.0

    # map each pitch to a position within its semitone [-50, +50) cents
    cents = 1200.0 * np.log2(voiced / F_REF)
    residuals = (cents % 100.0)          # 0..100
    residuals_centred = residuals - 50.0 # -50..+50

    # circular mean via angle representation
    angles = 2.0 * np.pi * residuals / 100.0
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    delta_t = (mean_angle / (2.0 * np.pi)) * 100.0   # back to cents in [-50, 50)
    return float(delta_t)


def compute_global_pitch_class_probability(
    spectrum: np.ndarray,
    fs: int = 44100,
    hop_size: int = 128,
    f_min: float = 120.0,
    f_max: float = 720.0,
    tuning_cents: float = 0.0,
) -> np.ndarray:
    """
    Compute global pitch class probability from averaged chroma (Eq. 15).

    spectrum: (n_frames, n_fft_bins) magnitude spectrum
    Returns L_global: shape (12,)
    """
    n_fft = (spectrum.shape[1] - 1) * 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # tuning-adjusted reference
    f_ref_tuned = F_REF * (2 ** (tuning_cents / 1200.0))

    chroma_acc = np.zeros(12, dtype=float)

    for frame in spectrum:
        chroma = np.zeros(12, dtype=float)
        for k, f in enumerate(freqs):
            if f < f_min or f > f_max:
                continue
            # map frequency to semitone bin
            semitone = 1200.0 * np.log2(f / f_ref_tuned) / 100.0
            chroma_bin = int(round(semitone)) % 12
            chroma[chroma_bin] += frame[k]
        chroma_acc += chroma

    total = chroma_acc.sum()
    if total < 1e-10:
        return np.ones(12) / 12.0
    return chroma_acc / total   # L_global, shape (12,)


def compute_local_pitch_probability(
    f0_segment_hz: np.ndarray,
    tuning_cents: float = 0.0,
    sigma_semitones: float = 0.5,  # quarter-tone spread (Eq. 17)
    n_semitones: int = 128,
) -> np.ndarray:
    """
    Build local pitch probability via Gaussian-smoothed histogram (Eq. 16-18).

    Returns L_local: shape (n_semitones,), indexed from MIDI 0.
    """
    f_ref_tuned = F_REF * (2 ** (tuning_cents / 1200.0))
    voiced = f0_segment_hz[f0_segment_hz > 0]
    if len(voiced) == 0:
        return np.zeros(n_semitones)

    cents = 1200.0 * np.log2(voiced / f_ref_tuned)
    # quantise to nearest semitone (MIDI-relative, centre at MIDI 69 = A4)
    semitone_indices = (np.round(cents / 100.0) + 69).astype(int)
    semitone_indices = np.clip(semitone_indices, 0, n_semitones - 1)

    # histogram H[ks]
    H = np.zeros(n_semitones, dtype=float)
    for ks in semitone_indices:
        H[ks] += 1.0
    H /= max(H.sum(), 1e-10)

    # Gaussian mixture: replace each bin with a Gaussian centred on it (Eq. 17-18)
    ks_axis = np.arange(n_semitones, dtype=float)
    L_local = np.zeros(n_semitones, dtype=float)
    for ks_prime in np.where(H > 0)[0]:
        gaussian = H[ks_prime] * np.exp(-0.5 * ((ks_axis - ks_prime) / sigma_semitones) ** 2)
        L_local += gaussian

    return L_local


def assign_pitch_label(
    f0_segment_hz: np.ndarray,
    L_global: np.ndarray,          # shape (12,)  from compute_global_pitch_class_probability
    tuning_cents: float = 0.0,
    n_semitones: int = 128,
) -> int:
    """
    Assign MIDI pitch to a note event (Eq. 19-21).

    L_pitch[ks] = L_global[ks mod 12] * L_local[ks]
    MIDI = 69 + argmax(L_pitch)
    """
    L_local = compute_local_pitch_probability(f0_segment_hz, tuning_cents, n_semitones=n_semitones)

    L_pitch = np.zeros(n_semitones, dtype=float)
    for ks in range(n_semitones):
        chroma_bin = ks % 12
        L_pitch[ks] = L_global[chroma_bin] * L_local[ks]

    if L_pitch.sum() < 1e-10:
        # fallback: median pitch
        voiced = f0_segment_hz[f0_segment_hz > 0]
        if len(voiced) == 0:
            return 69
        return int(round(69 + 12 * np.log2(np.median(voiced) / F_REF)))

    best_ks = int(np.argmax(L_pitch))
    # Eq. 21: P_midi = 69 + argmax(ks)  where ks is semitone offset from A4
    midi = best_ks   # L_local is indexed directly as MIDI number
    return int(np.clip(midi, 0, 127))


# ---------------------------------------------------------------------------
# II.B.3  Note post-processing
# ---------------------------------------------------------------------------

def post_process_notes(
    notes: List[Note],
    pitch_range_semitones: int = 8,     # ±8 around track median
    min_duration_s: float = 0.05,
) -> List[Note]:
    """
    Apply musicological constraints (Sec. II.B.3):
      1. Eliminate notes shorter than min_duration_s.
      2. Compute track median pitch; eliminate notes > 8 semitones below median.
      3. Transpose notes > 8 semitones above median down one octave.
    """
    if not notes:
        return notes

    # filter by minimum duration
    notes = [n for n in notes if n.duration >= min_duration_s]
    if not notes:
        return notes

    # compute median MIDI pitch
    pitches = np.array([n.midi for n in notes])
    median_pitch = float(np.median(pitches))

    filtered = []
    for note in notes:
        diff = note.midi - median_pitch
        if diff < -pitch_range_semitones:
            # likely guitar note – discard
            continue
        elif diff > pitch_range_semitones:
            # likely octave error – transpose down
            note = Note(note.onset, note.duration, note.midi - 12)
        filtered.append(note)

    return filtered


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def transcribe_notes(
    f0_hz: np.ndarray,
    rms: np.ndarray,
    spectrum: np.ndarray,
    fs: int = 44100,
    hop_size: int = 128,
    # segmentation params
    delta_p_min: float = 80.0,
    gauss_sigma_s: float = 0.0435,
    gauss_threshold: float = 4.0,
    volume_threshold_db: float = -10.0,
    pitch_z_threshold: float = -2.0,
    # post-processing params
    min_duration_s: float = 0.05,
    pitch_range_semitones: int = 8,
) -> List[Note]:
    """
    Full note transcription pipeline implementing Section II.B.

    Parameters
    ----------
    f0_hz       : (n_frames,)  vocal pitch contour in Hz; 0 = unvoiced
    rms         : (n_frames,)  frame-wise RMS amplitude
    spectrum    : (n_frames, n_fft_bins)  magnitude spectrum (rfft)
    fs          : sample rate in Hz
    hop_size    : analysis hop size in samples

    Returns
    -------
    List of Note(onset_s, duration_s, midi_pitch)
    """
    assert f0_hz.ndim == 1,           "f0_hz must be 1D"
    assert rms.ndim == 1,             "rms must be 1D"
    assert spectrum.ndim == 2,        "spectrum must be (n_frames, n_bins)"
    assert len(f0_hz) == len(rms) == len(spectrum), (
        f"Length mismatch: f0={len(f0_hz)}, rms={len(rms)}, spec={len(spectrum)}"
    )
    assert spectrum.shape[1] > 1,     "spectrum appears transposed"

    hop_s = hop_size / fs

    # --- tuning estimation (Eq. 14) ---
    tuning_cents = estimate_tuning_deviation(f0_hz)
    f_ref_tuned = F_REF * (2 ** (tuning_cents / 1200.0))

    # --- global pitch class probability (Eq. 15) ---
    L_global = compute_global_pitch_class_probability(
        spectrum, fs=fs, hop_size=hop_size, tuning_cents=tuning_cents
    )

    # --- note segmentation (Sec. II.B.1) ---
    # pre-filter: drop segments shorter than min_duration before labelling
    note_intervals = segment_contour(
        f0_hz, rms, fs=fs, hop_size=hop_size,
        delta_p_min=delta_p_min,
        gauss_sigma_s=gauss_sigma_s,
        gauss_threshold=gauss_threshold,
        volume_threshold_db=volume_threshold_db,
        pitch_z_threshold=pitch_z_threshold,
        min_gap_s=min_duration_s,
    )

    # --- pitch labelling (Sec. II.B.2) ---
    notes: List[Note] = []
    for seg_start, seg_end in note_intervals:
        seg_f0 = f0_hz[seg_start:seg_end]
        if (seg_f0 > 0).sum() == 0:
            continue

        onset_s = seg_start * hop_s
        duration_s = (seg_end - seg_start) * hop_s

        midi = assign_pitch_label(seg_f0, L_global, tuning_cents)
        notes.append(Note(onset=onset_s, duration=duration_s, midi=midi))

    # --- post-processing (Sec. II.B.3) ---
    notes = post_process_notes(notes, pitch_range_semitones, min_duration_s)

    return notes


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ---- synthetic demo data ----
    fs = 44100
    hop_size = 128
    hop_s = hop_size / fs
    duration_s = 10.0
    n_frames = int(duration_s / hop_s)

    rng = np.random.default_rng(0)

    # Simulate a voiced pitch contour: 3 notes with vibrato
    f0 = np.zeros(n_frames)
    note_freqs = [261.63, 293.66, 329.63]   # C4, D4, E4
    note_starts = [int(0.5 / hop_s), int(3.5 / hop_s), int(6.5 / hop_s)]
    note_ends   = [int(3.0 / hop_s), int(6.0 / hop_s), int(9.5 / hop_s)]

    t = np.arange(n_frames) * hop_s
    for freq, s, e in zip(note_freqs, note_starts, note_ends):
        vibrato = 1 + 0.005 * np.sin(2 * np.pi * 5.5 * t[s:e])
        f0[s:e] = freq * vibrato + rng.normal(0, 0.5, e - s)

    # Simulate RMS (slight dip at note boundaries)
    rms = 0.1 * np.ones(n_frames)
    for s in note_starts:
        rms[max(0, s - 3) : s + 3] *= 0.3

    # Simulate spectrum (random, used only for chroma)
    n_fft = 2048
    spectrum = np.abs(rng.normal(0, 0.1, (n_frames, n_fft // 2 + 1)))
    for freq, s, e in zip(note_freqs, note_starts, note_ends):
        k = int(round(freq * n_fft / fs))
        spectrum[s:e, k] += 1.0

    # ---- run transcription ----
    notes = transcribe_notes(f0, rms, spectrum, fs=fs, hop_size=hop_size)

    print("Transcribed notes:")
    print(f"{'Onset (s)':>10}  {'Duration (s)':>13}  {'MIDI':>5}  {'Note name':>9}")
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for n in notes:
        name = note_names[n.midi % 12] + str(n.midi // 12 - 1)
        print(f"{n.onset:>10.3f}  {n.duration:>13.3f}  {n.midi:>5}  {name:>9}")

    # ---- plot ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    t_frames = np.arange(n_frames) * hop_s

    axes[0].plot(t_frames, f0, lw=0.8, label="f0 (Hz)")
    for note in notes:
        axes[0].axvspan(note.onset, note.onset + note.duration, alpha=0.2, color="red")
        axes[0].text(note.onset + note.duration / 2, max(f0) * 0.95,
                     note_names[note.midi % 12], ha="center", fontsize=8)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Transcribed Notes (red shading)")
    axes[0].legend()

    axes[1].plot(t_frames, rms, lw=0.8, color="green", label="RMS")
    axes[1].set_ylabel("RMS")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/note_transcription_demo.png", dpi=150)
    print("\nPlot saved to note_transcription_demo.png")
