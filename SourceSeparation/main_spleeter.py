"""
Example: Spleeter source separation script
"""
import pandas as pd
from SourceSeparatorSpleeter import SpleeterSeparator
import time
from pathlib import Path

time_0 = time.time()

# Initialize Spleeter separator with 2 stems (vocals and accompaniment)
# You can use stems=4 or stems=5 for more detailed separation
separator = SpleeterSeparator(
    output_dir="spleeter_output",
    stems=2  # 2, 4, or 5 stems
)

# Process from folder
folder = Path("../data/cante2midiaudio")
wav_files = list(folder.glob("*.wav")) + list(folder.glob("*.mp3"))

# Process only if files exist
if wav_files:
    for audio_file in wav_files[:5]:  # Process first 5 files as example
        file_start_time = time.time()
        print(f"Processing {audio_file}")
        separator.separate_file(audio_file)
        print(f"File processed in {time.time()-file_start_time} seconds.")
else:
    print("No audio files found in the specified folder.")
    print("Please update the folder path to point to your audio files.")

# Alternative: to separate from df column
# df = pd.read_csv("wav_f0_paths_mapping.csv")  # contains column "path"
# separator.separate_from_dataframe(df[:2], "wav_path")

print(f"Total elapsed time: {time.time()-time_0} seconds")
