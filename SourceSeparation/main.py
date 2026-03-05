
import pandas as pd
from SourceSeparatorDemucs import DemucsSeparator
import torch
import time
from pathlib import Path

time_0 = time.time()

print(f"CUDA available? {torch.cuda.is_available()}")

separator = DemucsSeparator(
    output_dir="demucs_output",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

folder = Path("/home/ibroto/Documents/UPF_SMC/MIR/flamenco-mir/data/cante2midiaudio")
wav_files = list(folder.glob("*.wav"))

for audio_file in wav_files:
    file_start_time = time.time()
    print(f"Processing {audio_file}")
    separator.separate_file(audio_file)
    print(f"File processed in {time.time()-file_start_time} seconds.")


## to separate from df column
##df = pd.read_csv("/home/ibroto/Documents/UPF_SMC/MIR/flamenco-mir/SourceSeparation/wav_f0_paths_mapping.csv")  # contains column "path"
#separator.separate_from_dataframe(df[:2], "wav_path")

print(f"Ellapsed_time: {time.time()-time_0}")