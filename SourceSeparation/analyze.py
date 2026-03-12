"""
Alternative Evaluation Metrics for Source Separation
When ground truth separated stems are not available.

Includes:
- Spectral Centroids (voice vs accompaniment differences)
- Zero Crossing Rate
- MFCC-based similarity between original and reconstructed mix
- Frequency content analysis
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import pandas as pd
from scipy.signal import spectrogram
from scipy.fftpack import fft
import matplotlib.pyplot as plt


class SourceSeparationAnalyzer:
    """Analyze source separation without ground truth reference."""
    
    def __init__(self, mixture_dir: Path, output_dir: Path):
        """
        Initialize analyzer.
        
        Args:
            mixture_dir: path to original mixed audio
            output_dir: path to separated stems
        """
        self.mixture_dir = Path(mixture_dir)
        self.output_dir = Path(output_dir)
        self.results = []
    
    @staticmethod
    def load_audio(file_path: Path, mono=True):
        """Load audio file."""
        try:
            audio, sr = sf.read(file_path)
            if mono and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    @staticmethod
    def spectral_centroid(audio, sr):
        """Compute spectral centroid (higher for bright sounds like vocals)."""
        # FFT
        fft_result = np.abs(fft(audio))
        freqs = np.fft.fftfreq(len(audio), 1/sr)[:len(audio)//2]
        magnitude = fft_result[:len(audio)//2]
        
        centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        return centroid
    
    @staticmethod
    def zero_crossing_rate(audio):
        """Compute zero crossing rate (higher for vocals/noise)."""
        zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2
        return zcr
    
    @staticmethod
    def spectral_flux(audio, sr, hop_length=512):
        """Compute spectral flux (measures spectral change)."""
        f, t, Sxx = spectrogram(audio, sr, nperseg=2048, hoplen=hop_length)
        
        # Compute differences between consecutive frames
        spectral_diffs = np.sqrt(np.sum(np.diff(Sxx, axis=1)**2, axis=0))
        
        return np.mean(spectral_diffs), np.std(spectral_diffs)
    
    @staticmethod
    def compute_energy(audio):
        """Compute RMS energy."""
        return np.sqrt(np.mean(audio**2))
    
    def analyze_track(self, track_name: str):
        """
        Analyze a single track.
        
        Args:
            track_name: name of the track
            
        Returns:
            dict with analysis metrics
        """
        result = {'track': track_name}
        
        # Load original mixture
        mixture_path = self.mixture_dir / f"{track_name}.wav"
        if not mixture_path.exists():
            # Try with different extensions
            for ext in ['.mp3', '.flac', '.ogg']:
                alt_path = self.mixture_dir / f"{track_name}{ext}"
                if alt_path.exists():
                    mixture_path = alt_path
                    break
        
        mixture, sr = self.load_audio(mixture_path)
        if mixture is None:
            print(f"Mixture not found: {mixture_path}")
            return None
        
        # Load separated stems
        vocal_path = self.output_dir / track_name / "vocals.wav"
        accomp_path = self.output_dir / track_name / "accompaniment.wav"
        
        vocal, _ = self.load_audio(vocal_path)
        accomp, _ = self.load_audio(accomp_path)
        
        if vocal is None or accomp is None:
            print(f"Stems not found for {track_name}")
            return None
        
        # Ensure same length
        min_len = min(len(vocal), len(accomp), len(mixture))
        vocal = vocal[:min_len]
        accomp = accomp[:min_len]
        mixture = mixture[:min_len]
        
        # Analysis: Spectral properties
        result['mixture_centroid'] = self.spectral_centroid(mixture, sr)
        result['vocal_centroid'] = self.spectral_centroid(vocal, sr)
        result['accomp_centroid'] = self.spectral_centroid(accomp, sr)
        result['centroid_diff'] = abs(result['vocal_centroid'] - result['accomp_centroid'])
        
        # Analysis: Zero crossing rate (voice is usually higher)
        result['mixture_zcr'] = self.zero_crossing_rate(mixture)
        result['vocal_zcr'] = self.zero_crossing_rate(vocal)
        result['accomp_zcr'] = self.zero_crossing_rate(accomp)
        
        # Analysis: Energy
        result['mixture_energy'] = self.compute_energy(mixture)
        result['vocal_energy'] = self.compute_energy(vocal)
        result['accomp_energy'] = self.compute_energy(accomp)
        result['energy_vocal_ratio'] = result['vocal_energy'] / (result['mixture_energy'] + 1e-12)
        
        # Analysis: Spectral flux
        mix_flux, mix_flux_std = self.spectral_flux(mixture, sr)
        vocal_flux, vocal_flux_std = self.spectral_flux(vocal, sr)
        accomp_flux, accomp_flux_std = self.spectral_flux(accomp, sr)
        
        result['mixture_flux'] = mix_flux
        result['vocal_flux'] = vocal_flux
        result['accomp_flux'] = accomp_flux
        
        # Analysis: Reconstruction (mix should ≈ vocals + accompaniment)
        reconstructed = vocal + accomp
        reconstruction_error = np.mean((mixture - reconstructed)**2)
        reconstruction_snr = 10 * np.log10(np.mean(mixture**2) / (reconstruction_error + 1e-12))
        
        result['reconstruction_snr'] = reconstruction_snr
        result['reconstruction_error'] = reconstruction_error
        
        return result
    
    def analyze_all(self):
        """Analyze all separated tracks."""
        self.results = []
        
        track_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        
        print(f"\nAnalyzing {len(track_dirs)} tracks...\n")
        
        for idx, track_dir in enumerate(sorted(track_dirs), 1):
            track_name = track_dir.name
            print(f"[{idx}/{len(track_dirs)}] Analyzing {track_name}...")
            
            result = self.analyze_track(track_name)
            if result:
                self.results.append(result)
        
        return pd.DataFrame(self.results)
    
    def print_summary(self, df: pd.DataFrame):
        """Print analysis summary."""
        print("\n" + "=" * 80)
        print("SOURCE SEPARATION ANALYSIS SUMMARY")
        print("=" * 80)
        
        print("\nSpectral Analysis:")
        print("-" * 80)
        print(f"Vocal Spectral Centroid:  {df['vocal_centroid'].mean():10.0f} Hz (mean)")
        print(f"Accomp Spectral Centroid: {df['accomp_centroid'].mean():10.0f} Hz (mean)")
        print(f"Centroid Difference:      {df['centroid_diff'].mean():10.0f} Hz (mean)")
        print(f"  → Higher is better (more clear separation)")
        
        print("\nZero Crossing Rate:")
        print("-" * 80)
        print(f"Vocal ZCR:    {df['vocal_zcr'].mean():10.4f} (higher = more detail)")
        print(f"Accomp ZCR:   {df['accomp_zcr'].mean():10.4f}")
        
        print("\nEnergy Distribution:")
        print("-" * 80)
        print(f"Vocal Energy Ratio: {df['energy_vocal_ratio'].mean():6.2%}")
        print(f"Accomp Energy:      {df['accomp_energy'].mean():10.2f} (mean)")
        
        print("\nReconstruction Quality:")
        print("-" * 80)
        print(f"SNR (vocals + accomp vs original): {df['reconstruction_snr'].mean():6.2f} dB (mean)")
        print(f"  → Higher is better (< 20dB good, < 12dB acceptable)")
        
        print("\nDetailed Results:")
        print("-" * 80)
        display_cols = ['track', 'vocal_centroid', 'accomp_centroid', 'centroid_diff', 
                       'energy_vocal_ratio', 'reconstruction_snr']
        display_df = df[display_cols].round(2)
        print(display_df.to_string(index=False))
        print("\n" + "=" * 80)
    
    def save_results(self, output_path: Path):
        """Save analysis to CSV."""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(output_path, index=False)
            print(f"\n✅ Analysis saved to {output_path}")
            return df
        else:
            print("No results to save")
            return None


def main():
    """Run analysis without ground truth."""
    
    project_root = Path(__file__).parent.parent
    mixture_dir = project_root / "data" / "cante2midiaudio"
    output_dir = project_root / "SourceSeparation" / "spleeter_output"
    
    analyzer = SourceSeparationAnalyzer(
        mixture_dir=mixture_dir,
        output_dir=output_dir
    )
    
    print("Starting Source Separation Analysis...")
    print(f"(No ground truth required)\n")
    
    # Analyze
    results_df = analyzer.analyze_all()
    
    # Print summary
    if not results_df.empty:
        analyzer.print_summary(results_df)
        
        # Save
        output_csv = output_dir / "analysis_results.csv"
        analyzer.save_results(output_csv)


if __name__ == "__main__":
    main()
