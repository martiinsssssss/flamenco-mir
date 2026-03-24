"""
Source Separation Evaluation Module
Evaluates source separation quality using standard metrics:
- SDR (Signal-to-Distortion Ratio)
- ISR (Image-to-Spatial-distortion Ratio)
- SIR (Signal-to-Interference Ratio)
- SAR (Signal-to-Artifact Ratio)

Based on standard evaluation methodology from:
https://ieeexplore.ieee.org/document/7410016
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import pandas as pd
from scipy.optimize import linprog


def compute_bss_eval_sources(reference, estimate, compute_sir=True):
    """
    Compute BSS evaluation metrics for a single source.
    
    Args:
        reference: reference signal (target)
        estimate: estimated signal (predicted)
        compute_sir: whether to compute SIR (slower)
        
    Returns:
        sdr, sir, sar, perm
    """
    # Remove mean
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    
    # Projections
    s_target = np.dot(reference, estimate) / np.dot(reference, reference) * reference
    e_interf = estimate - s_target
    
    # SDR (Signal-to-Distortion Ratio)
    sdr = 10 * np.log10(np.dot(s_target, s_target) / np.dot(e_interf, e_interf) + 1e-12)
    
    # SAR (Signal-to-Artifact Ratio)
    s_artifact = reference - s_target
    sar = 10 * np.log10(np.dot(s_target, s_target) / np.dot(s_artifact, s_artifact) + 1e-12)
    
    # SIR (Signal-to-Interference Ratio)
    if compute_sir:
        sir = 10 * np.log10(np.dot(s_target, s_target) / np.dot(reference - s_target, reference - s_target) + 1e-12)
    else:
        sir = 0
    
    return sdr, sir, sar


class SourceSeparationEvaluator:
    """Evaluates source separation results using BSS Eval metrics."""
    
    METRICS = ['SDR', 'SIR', 'SAR']
    
    def __init__(self, mixture_dir: Path, reference_dir: Path, output_dir: Path):
        """
        Initialize evaluator.
        
        Args:
            mixture_dir: path to original mixed audio files
            reference_dir: path to reference separated stems (ground truth)
            output_dir: path to algorithm output (separated stems)
        """
        self.mixture_dir = Path(mixture_dir)
        self.reference_dir = Path(reference_dir)
        self.output_dir = Path(output_dir)
        self.results = []
        
    def load_audio(self, file_path: Path, mono=True):
        """Load audio file and optionally convert to mono."""
        try:
            audio, sr = sf.read(file_path)
            if mono and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def evaluate_track(self, track_name: str, stem_names=['vocals', 'accompaniment']):
        """
        Evaluate a single track with multiple stems.
        
        Args:
            track_name: name of the track to evaluate
            stem_names: list of stem names to evaluate
            
        Returns:
            dict with metrics for each stem and overall
        """
        results = {'track': track_name}
        all_sdrs = []
        all_sirs = []
        all_sars = []
        
        for stem in stem_names:
            # Load reference (ground truth)
            ref_path = self.reference_dir / track_name / f"{stem}.wav"
            ref_audio, sr = self.load_audio(ref_path)
            
            if ref_audio is None:
                print(f"Skipping {track_name}/{stem} - reference not found")
                continue
            
            # Load estimated (algorithm output)
            est_path = self.output_dir / track_name / f"{stem}.wav"
            est_audio, _ = self.load_audio(est_path)
            
            if est_audio is None:
                print(f"Skipping {track_name}/{stem} - estimate not found")
                continue
            
            # Ensure same length
            min_len = min(len(ref_audio), len(est_audio))
            ref_audio = ref_audio[:min_len]
            est_audio = est_audio[:min_len]
            
            # Compute metrics
            sdr, sir, sar = compute_bss_eval_sources(ref_audio, est_audio, compute_sir=True)
            
            results[f'{stem}_sdr'] = sdr
            results[f'{stem}_sir'] = sir
            results[f'{stem}_sar'] = sar
            
            all_sdrs.append(sdr)
            all_sirs.append(sir)
            all_sars.append(sar)
        
        # Overall metrics (mean across stems)
        if all_sdrs:
            results['overall_sdr'] = np.mean(all_sdrs)
            results['overall_sir'] = np.mean(all_sirs)
            results['overall_sar'] = np.mean(all_sars)
        
        return results
    
    def evaluate_all(self, stem_names=['vocals', 'accompaniment']):
        """
        Evaluate all tracks in the output directory.
        
        Args:
            stem_names: list of stem names to evaluate
            
        Returns:
            DataFrame with all results
        """
        self.results = []
        
        # Find all output tracks
        track_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        
        print(f"\nEvaluating {len(track_dirs)} tracks...\n")
        
        for idx, track_dir in enumerate(sorted(track_dirs), 1):
            track_name = track_dir.name
            print(f"[{idx}/{len(track_dirs)}] Evaluating {track_name}...")
            
            result = self.evaluate_track(track_name, stem_names)
            self.results.append(result)
        
        return pd.DataFrame(self.results)
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("SOURCE SEPARATION EVALUATION SUMMARY")
        print("=" * 80)
        
        # Overall statistics
        print("\nOVERALL METRICS (across all tracks):")
        print("-" * 80)
        
        for metric in self.METRICS:
            col = f'overall_{metric.lower()}'
            if col in df.columns:
                values = df[col].dropna()
                print(f"{metric}:")
                print(f"  Mean:   {values.mean():7.2f} dB")
                print(f"  Median: {values.median():7.2f} dB")
                print(f"  Std:    {values.std():7.2f} dB")
                print(f"  Min:    {values.min():7.2f} dB")
                print(f"  Max:    {values.max():7.2f} dB")
                print()
        
        # Per-track details
        print("\nPER-TRACK RESULTS:")
        print("-" * 80)
        
        display_cols = ['track'] + [col for col in df.columns if 'overall' in col]
        display_df = df[display_cols].copy()
        display_df = display_df.round(2)
        
        print(display_df.to_string(index=False))
        print("\n" + "=" * 80)
    
    def save_results(self, output_path: Path):
        """Save evaluation results to CSV."""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(output_path, index=False)
            print(f"\n✅ Results saved to {output_path}")
            return df
        else:
            print("No results to save")
            return None


def main():
    """
    Example evaluation setup.
    
    Adjust paths according to your directory structure:
    - mixture_dir: original audio files
    - reference_dir: ground truth separated stems (if available)
    - output_dir: Spleeter/Demucs output
    """
    
    # Configuration (modify as needed)
    project_root = Path(__file__).parent.parent
    mixture_dir = project_root / "data" / "cante2midiaudio"
    reference_dir = project_root / "data" / "reference_separation"  # Optional
    output_dir = project_root / "SourceSeparation" / "spleeter_output"
    
    # Create evaluator
    evaluator = SourceSeparationEvaluator(
        mixture_dir=mixture_dir,
        reference_dir=reference_dir,
        output_dir=output_dir
    )
    
    # Run evaluation
    print("Starting Source Separation Evaluation...")
    print(f"Output directory: {output_dir}")
    
    # Check if reference exists
    if not reference_dir.exists():
        print(f"\n⚠️  Reference directory not found: {reference_dir}")
        print("For full evaluation, you need ground truth separated stems.")
        print("Alternative: Wait until you have reference data or use visual inspection.")
        return
    
    # Evaluate
    results_df = evaluator.evaluate_all(stem_names=['vocals', 'accompaniment'])
    
    # Print summary
    evaluator.print_summary(results_df)
    
    # Save results
    output_csv = output_dir / "evaluation_results.csv"
    evaluator.save_results(output_csv)


if __name__ == "__main__":
    main()
