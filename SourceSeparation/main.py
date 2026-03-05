
import pandas as pd
from SourceSeparator import DemucsSeparator
import torch
import time

time_0 = time.time()

df = pd.read_csv("/home/ibroto/Documents/UPF_SMC/MIR/flamenco-mir/SourceSeparation/wav_f0_paths_mapping.csv")  # contains column "path"

print(f"CUDA available? {torch.cuda.is_available()}")

separator = DemucsSeparator(
    output_dir="demucs_output",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

separator.separate_from_dataframe(df[:2], "wav_path")


print(f"Ellapsed_time: {time.time()-time_0}")