from pesto_tracker import PESTOTracker
from pathlib import Path



if __name__ == "__main__":

    folder = Path("/home/ibroto/Documents/UPF_SMC/MIR/flamenco-mir/SourceSeparation/demucs_output")
    wav_files = list(folder.glob("**/voice.wav"))
    print(f"{len(wav_files)} wav files found.")
    tracker = PESTOTracker(
        confidence_threshold=0.5,
        step_size=0.01,
        use_gpu=True
    )
    for file in wav_files[:2]:
        pitches, times, confidence = tracker.extract_f0(file)
        print(f"Extracted {len(pitches)} pitch frames")
        print(f"Voiced frames: {(confidence > 0.5).sum()}")


