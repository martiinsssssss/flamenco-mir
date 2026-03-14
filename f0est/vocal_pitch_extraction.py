"""
Vocal Pitch Extraction - Section II.A
Kroher & Gomez, "Automatic Transcription of Flamenco Singing from Polyphonic Music Recordings"

Implements:
  II.A.1 - Channel selection (spectral band ratio, Eq. 1-2)
  II.A.2 - Predominant melody extraction (wrapper around melodia / your f0 extractor)
  II.A.3 - Contour filtering (bark band Gaussian classifier + binary moving average, Eq. 3-6)

Inputs assumed available per track:
  - wav_path         : path to stereo (or mono) WAV file
  - bark_bands[n,12] : energy in lower 12 bark bands, one row per frame
  - f0_hz[n]         : predominant melody output in Hz (0 = non-melody frame)
  - fs               : sample rate (44100)
  - hop_size         : hop size used for f0 / bark band extraction (128)
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from typing import Tuple, Optional
import warnings


# ---------------------------------------------------------------------------
# II.A.1  Channel selection
# ---------------------------------------------------------------------------

def compute_spectral_band_ratio(
    audio_channel: np.ndarray,
    fs: int = 44100,
    n_fft: int = 4096,
    hop_size: int = 1024,
    zero_pad: int = 2,
    f_low: Tuple[float, float] = (80.0, 400.0),    # lower band Hz limits
    f_high: Tuple[float, float] = (500.0, 6000.0), # upper band Hz limits
) -> float:
    """
    Compute average spectral band ratio S for one audio channel (Eq. 1-2).

    S[n] = 20*log10( sum(|X_norm[k]| for k in upper_band)
                   / sum(|X_norm[k]| for k in lower_band) )

    The magnitude spectrum is normalised by its per-frame maximum before
    summing so that overall volume does not bias the ratio.

    Returns the average S over all frames.
    """
    n_fft_padded = n_fft * zero_pad
    n_frames = 1 + (len(audio_channel) - n_fft) // hop_size

    # frequency axis for the padded FFT
    freqs = np.fft.rfftfreq(n_fft_padded, d=1.0 / fs)

    # bin ranges  (Eq. 1)
    def freq_bin(f):
        return int(round(f * zero_pad * n_fft / fs))

    k_l1, k_l2 = freq_bin(f_low[0]),  freq_bin(f_low[1])
    k_h1, k_h2 = freq_bin(f_high[0]), freq_bin(f_high[1])

    ratios = []
    for i in range(n_frames):
        frame = audio_channel[i * hop_size : i * hop_size + n_fft]
        if len(frame) < n_fft:
            break
        frame = frame * np.hanning(n_fft)
        X = np.abs(np.fft.rfft(frame, n=n_fft_padded))

        # normalise by frame maximum (Eq. 2)
        X_norm = X / (X.max() + 1e-10)

        sum_high = X_norm[k_h1:k_h2].sum()
        sum_low  = X_norm[k_l1:k_l2].sum()

        if sum_low < 1e-10:
            continue
        ratios.append(20.0 * np.log10(sum_high / sum_low + 1e-10))

    return float(np.mean(ratios)) if ratios else 0.0


def select_dominant_vocal_channel(
    audio: np.ndarray,
    fs: int = 44100,
    n_fft: int = 4096,
    hop_size: int = 1024,
) -> Tuple[np.ndarray, int]:
    """
    Select the stereo channel with stronger vocal presence (Sec. II.A.1).

    Parameters
    ----------
    audio : (n_samples,) mono  OR  (n_samples, 2) / (2, n_samples) stereo

    Returns
    -------
    selected_channel : 1-D array of the chosen channel
    channel_idx      : 0 (left) or 1 (right); -1 if mono
    """
    # --- normalise shape to (n_samples, n_channels) ---
    if audio.ndim == 1:
        return audio, -1                     # mono: nothing to choose

    if audio.shape[0] == 2 and audio.shape[1] != 2:
        audio = audio.T                      # (2, N) -> (N, 2)

    if audio.shape[1] == 1:
        return audio[:, 0], -1              # effectively mono

    left  = audio[:, 0].astype(float)
    right = audio[:, 1].astype(float)

    s_left  = compute_spectral_band_ratio(left,  fs, n_fft, hop_size)
    s_right = compute_spectral_band_ratio(right, fs, n_fft, hop_size)

    if s_left >= s_right:
        return left, 0
    else:
        return right, 1


# ---------------------------------------------------------------------------
# II.A.2  Predominant melody extraction (interface / wrapper)
# ---------------------------------------------------------------------------

def extract_predominant_melody(
    audio_channel: np.ndarray,
    fs: int = 44100,
    hop_size: int = 128,
    f_min: float = 120.0,
    f_max: float = 720.0,
    voicing_tolerance: float = 0.2,
    method: str = "melodia",
) -> np.ndarray:
    """
    Extract the predominant melody pitch contour (Sec. II.A.2).

    The paper uses the Salamon & Gomez algorithm [26] (MELODIA), available
    in essentia. We wrap it here and provide a librosa-based fallback (pyin)
    so the code runs without essentia installed.

    Parameters
    ----------
    audio_channel     : 1-D mono signal
    fs                : sample rate
    hop_size          : hop size in samples (paper uses 128)
    f_min / f_max     : expected vocal pitch range in Hz (paper: 120–720 Hz)
    voicing_tolerance : salience threshold (paper: τ_v = 0.2 for polyphonic,
                        3.0 for monophonic). Lower = more frames kept.
    method            : "melodia"  use essentia's PredominantPitchMelodia
                        "pyin"     use librosa's pyin as fallback

    Returns
    -------
    f0_hz : (n_frames,) pitch in Hz; 0.0 for non-melody frames
    """
    if method == "melodia":
        return _melodia(audio_channel, fs, hop_size, f_min, f_max, voicing_tolerance)
    elif method == "pyin":
        return _pyin(audio_channel, fs, hop_size, f_min, f_max)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'melodia' or 'pyin'.")


def _melodia(audio, fs, hop_size, f_min, f_max, voicing_tolerance):
    """Essentia PredominantPitchMelodia wrapper."""
    try:
        import essentia.standard as es
    except ImportError:
        warnings.warn(
            "essentia not installed. Falling back to pyin. "
            "Install with: pip install essentia",
            stacklevel=3,
        )
        return _pyin(audio, fs, hop_size, f_min, f_max)

    extractor = es.PredominantPitchMelodia(
        frameSize=4096,
        hopSize=hop_size,
        sampleRate=fs,
        minFrequency=f_min,
        maxFrequency=f_max,
        voicingTolerance=voicing_tolerance,
        guessUnvoiced=False,
    )
    pitch, pitch_confidence = extractor(audio.astype(np.float32))
    f0_hz = np.array(pitch, dtype=float)
    # essentia returns 0 for unvoiced frames already
    return f0_hz


def _pyin(audio, fs, hop_size, f_min, f_max):
    """librosa pyin fallback."""
    try:
        import librosa
    except ImportError:
        raise ImportError("Neither essentia nor librosa is available.")

    f0, voiced_flag, _ = librosa.pyin(
        audio.astype(float),
        fmin=f_min,
        fmax=f_max,
        sr=fs,
        hop_length=hop_size,
        fill_na=0.0,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    f0[~voiced_flag] = 0.0
    return f0.astype(float)


# ---------------------------------------------------------------------------
# II.A.3  Contour filtering
# ---------------------------------------------------------------------------

def fit_gaussian(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a multivariate Gaussian to a set of feature vectors.

    Returns (mean, covariance).  If degenerate, adds small regularisation.
    """
    mu = features.mean(axis=0)
    cov = np.cov(features, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    # regularise to avoid singular covariance
    cov += np.eye(cov.shape[0]) * 1e-6
    return mu, cov


def gaussian_log_likelihood(
    x: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
) -> float:
    """
    Log-likelihood of feature vector x under N(mu, cov).  (Eq. 4)

    Using log-space for numerical stability instead of raw probability.
    """
    d = x - mu
    sign, log_det = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf
    inv_cov = np.linalg.inv(cov)
    maha = float(d @ inv_cov @ d)
    k = len(mu)
    return -0.5 * (k * np.log(2 * np.pi) + log_det + maha)


def classify_vocal_frames(
    bark_bands: np.ndarray,
    f0_hz: np.ndarray,
    moving_avg_s: float = 1.0,
    fs: int = 44100,
    hop_size: int = 128,
) -> np.ndarray:
    """
    Frame-wise vocal / non-vocal classification (Sec. II.A.3, Eq. 3-5).

    Uses the lower 12 bark band energies.  The initial voiced/unvoiced
    labelling from the predominant melody algorithm is used to fit two
    Gaussian distributions (voiced x+ and unvoiced x-).  Each frame is
    then re-classified by comparing likelihoods.  A 1-second moving
    average filter smooths the result.

    Parameters
    ----------
    bark_bands   : (n_frames, 12)  energy in the lower 12 bark bands (Eq. 3)
    f0_hz        : (n_frames,)     raw predominant melody (0 = non-melody)
    moving_avg_s : length of binary moving average filter in seconds
    fs, hop_size : for converting seconds to frames

    Returns
    -------
    v : (n_frames,) binary array; 1 = vocal, 0 = non-vocal
    """
    assert bark_bands.shape[1] >= 12, \
        f"Expected at least 12 bark bands, got {bark_bands.shape[1]}"
    assert len(bark_bands) == len(f0_hz), \
        f"Length mismatch: bark_bands={len(bark_bands)}, f0={len(f0_hz)}"

    features = bark_bands.iloc[:, :12].astype(float)

    # --- initial labelling from predominant melody output ---
    voiced_mask   = (f0_hz > 0)
    unvoiced_mask = ~voiced_mask

    n_voiced   = voiced_mask.sum()
    n_unvoiced = unvoiced_mask.sum()

    if n_voiced < 13:
        warnings.warn("Too few voiced frames to fit Gaussian; returning raw voicing.")
        return voiced_mask.astype(int)

    if n_unvoiced < 13:
        warnings.warn("Too few unvoiced frames to fit Gaussian; returning raw voicing.")
        return voiced_mask.astype(int)

    # --- fit Gaussians (Eq. 4) ---
    mu_pos, cov_pos = fit_gaussian(features[voiced_mask])
    mu_neg, cov_neg = fit_gaussian(features[unvoiced_mask])

    # --- frame-wise classification (Eq. 5) ---
    v_raw = np.zeros(len(features), dtype=float)
    for i, x in enumerate(features):
        ll_pos = gaussian_log_likelihood(x, mu_pos, cov_pos)
        ll_neg = gaussian_log_likelihood(x, mu_neg, cov_neg)
        v_raw[i] = 1.0 if ll_pos >= ll_neg else 0.0

    # --- 1-second binary moving average filter ---
    filter_frames = max(1, int(moving_avg_s * fs / hop_size))
    v_smooth = uniform_filter1d(v_raw, size=filter_frames, mode="nearest")

    # threshold at 0.5 to recover binary prediction (Eq. 5 / Fig. 3c)
    v = (v_smooth >= 0.5).astype(int)
    return v


def filter_guitar_contours(
    f0_hz: np.ndarray,
    v: np.ndarray,
    hop_size: int = 128,
    fs: int = 44100,
) -> np.ndarray:
    """
    Eliminate contour segments that lie entirely outside vocal regions (Eq. 6).

    A contour = maximal run of consecutive melody frames (f0 > 0).
    A contour is eliminated if the sum of v over it equals 0.

    Returns
    -------
    f0_vocal : copy of f0_hz with guitar contours zeroed out
    """
    assert len(f0_hz) == len(v), "f0_hz and v must have the same length"

    f0_vocal = f0_hz.copy().astype(float)

    # find contiguous melody segments
    is_melody = (f0_hz > 0).astype(int)
    diff = np.diff(np.concatenate(([0], is_melody, [0])))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        # Eq. 6: eliminate if entirely outside vocal regions
        if v[s:e].sum() == 0:
            f0_vocal[s:e] = 0.0

    return f0_vocal


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def extract_vocal_pitch(
    audio: np.ndarray,
    bark_bands: np.ndarray,
    fs: int = 44100,
    hop_size: int = 128,
    f_min: float = 120.0,
    f_max: float = 720.0,
    voicing_tolerance: float = 0.2,
    melody_method: str = "melodia",
    moving_avg_s: float = 1.0,
    channel_selection_hop: int = 1024,
    f0_hz_precomputed: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Full vocal pitch extraction pipeline (Section II.A).

    Parameters
    ----------
    audio            : (n_samples,) mono or (n_samples, 2) stereo WAV signal
    bark_bands       : (n_frames, >=12)  lower 12 bark band energies (Eq. 3)
                       must be computed at the same hop_size as f0
    fs               : sample rate
    hop_size         : hop size for f0 / bark bands (paper: 128)
    f_min / f_max    : vocal pitch range in Hz
    voicing_tolerance: melodia voicing threshold (0.2 polyphonic, 3.0 mono)
    melody_method    : "melodia" or "pyin"
    moving_avg_s     : length of smoothing filter for vocal classifier (1 s)
    channel_selection_hop : hop size used only for channel selection (1024)
    f0_hz_precomputed: if you already have f0, pass it here to skip step II.A.2

    Returns
    -------
    f0_vocal   : (n_frames,)  vocal pitch in Hz; 0 for non-vocal / unvoiced
    v          : (n_frames,)  binary vocal mask (1=vocal, 0=guitar/silence)
    channel    : selected stereo channel index (0=left, 1=right, -1=mono)
    """

    # ------------------------------------------------------------------
    # Step 1 - Channel selection (Sec. II.A.1)
    # ------------------------------------------------------------------
    selected_audio, channel_idx = select_dominant_vocal_channel(
        audio, fs=fs, hop_size=channel_selection_hop
    )

    # ------------------------------------------------------------------
    # Step 2 - Predominant melody extraction (Sec. II.A.2)
    # ------------------------------------------------------------------
    if f0_hz_precomputed is not None:
        f0_raw = np.nan_to_num(f0_hz_precomputed, nan=0.0).astype(float)
    else:
        f0_raw = extract_predominant_melody(
            selected_audio,
            fs=fs,
            hop_size=hop_size,
            f_min=f_min,
            f_max=f_max,
            voicing_tolerance=voicing_tolerance,
            method=melody_method,
        )

    # align lengths (bark_bands and f0 might differ by ±1 frame)
    n = min(len(f0_raw), len(bark_bands))
    f0_raw    = f0_raw[:n]
    bark_use  = bark_bands[:n]

    # ------------------------------------------------------------------
    # Step 3 - Contour filtering (Sec. II.A.3)
    # ------------------------------------------------------------------
    v = classify_vocal_frames(
        bark_use, f0_raw,
        moving_avg_s=moving_avg_s,
        fs=fs,
        hop_size=hop_size,
    )

    f0_vocal = filter_guitar_contours(f0_raw, v, hop_size=hop_size, fs=fs)

    return f0_vocal, v, channel_idx


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_audio(wav_path: str, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file, keeping stereo if present.

    Returns (audio, fs) where audio is (n_samples,) or (n_samples, 2).
    """
    try:
        import soundfile as sf
        audio, fs = sf.read(wav_path, always_2d=False)
    except ImportError:
        try:
            import librosa
            audio, fs = librosa.load(wav_path, sr=target_sr, mono=False)
            if audio.ndim == 2:
                audio = audio.T          # librosa returns (channels, samples)
        except ImportError:
            raise ImportError("Install soundfile or librosa to load audio.")
    return audio, fs


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    fs = 44100
    hop_size = 128
    duration_s = 15.0
    n_samples = int(duration_s * fs)
    n_frames  = int(duration_s * fs / hop_size)

    # --- simulate stereo audio: vocals stronger in left channel ---
    t = np.arange(n_samples) / fs
    # vocal sections: 0-4s, 7-11s, 13-15s
    vocal_mask_audio = (
        ((t >= 0)  & (t < 4)) |
        ((t >= 7)  & (t < 11)) |
        ((t >= 13) & (t < 15))
    )
    vocal_signal = np.where(vocal_mask_audio,
                            np.sin(2 * np.pi * 300 * t) * 0.8, 0.0)
    guitar_signal = np.sin(2 * np.pi * 200 * t) * 0.3

    left  = vocal_signal + guitar_signal * 0.2 + rng.normal(0, 0.01, n_samples)
    right = guitar_signal              + rng.normal(0, 0.01, n_samples)
    audio = np.stack([left, right], axis=1)

    # --- simulate bark bands (12 bands) ---
    # vocal sections have more energy in upper bands
    frame_times = np.arange(n_frames) * hop_size / fs
    vocal_frame_mask = (
        ((frame_times >= 0)  & (frame_times < 4)) |
        ((frame_times >= 7)  & (frame_times < 11)) |
        ((frame_times >= 13) & (frame_times < 15))
    )

    bark_bands = rng.uniform(0.0, 0.1, (n_frames, 12))
    bark_bands[vocal_frame_mask,  6:] += 0.5   # vocal: more energy in upper bands
    bark_bands[~vocal_frame_mask, :6] += 0.5   # guitar: more energy in lower bands

    # --- simulate f0 (precomputed) ---
    f0_hz = np.zeros(n_frames)
    vocal_freq = 300.0
    guitar_freq = 200.0
    f0_hz[vocal_frame_mask]  = vocal_freq  * (1 + 0.004 * np.sin(2 * np.pi * 5 * frame_times[vocal_frame_mask]))
    f0_hz[~vocal_frame_mask] = guitar_freq * (1 + 0.003 * np.sin(2 * np.pi * 3 * frame_times[~vocal_frame_mask]))

    # --- run pipeline ---
    f0_vocal, v, channel = extract_vocal_pitch(
        audio=audio,
        bark_bands=bark_bands,
        fs=fs,
        hop_size=hop_size,
        f0_hz_precomputed=f0_hz,
        moving_avg_s=1.0,
    )

    print(f"Selected channel: {'left' if channel == 0 else 'right' if channel == 1 else 'mono'}")
    print(f"Frames total:  {n_frames}")
    print(f"Frames vocal (ground truth): {vocal_frame_mask.sum()}")
    print(f"Frames vocal (predicted):    {v.sum()}")
    print(f"Frames with f0 after filtering: {(f0_vocal > 0).sum()}")

    # --- plot ---
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)

    axes[0].plot(frame_times, f0_hz, lw=0.7, color="gray", label="raw f0")
    axes[0].plot(frame_times, np.where(f0_vocal > 0, f0_vocal, np.nan),
                 lw=1.2, color="red", label="vocal f0")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Vocal Pitch Extraction (Section II.A)")
    axes[0].legend(fontsize=8)

    axes[1].imshow(bark_bands.T, aspect="auto", origin="lower",
                   extent=[0, duration_s, 0, 12])
    axes[1].set_ylabel("Bark band")
    axes[1].set_title("Bark band energies")

    axes[2].fill_between(frame_times, vocal_frame_mask.astype(float),
                         alpha=0.4, color="green", label="ground truth vocal")
    axes[2].plot(frame_times, v, lw=1.2, color="red", label="predicted vocal")
    axes[2].set_ylabel("Vocal mask")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Vocal / non-vocal classification")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/vocal_pitch_extraction_demo.png", dpi=150)
    print("\nPlot saved.")
