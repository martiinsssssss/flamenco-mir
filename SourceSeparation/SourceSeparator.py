import os
from pathlib import Path
##Avoid TorchCodec dependancies
#import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import soundfile as sf
import torch
import time

class DemucsSeparator:
    def __init__(self, output_dir="separated", device="cuda"):
        """
        Simple Demucs wrapper for batch source separation.

        Args
        ----
        output_dir : str
            Where separated stems will be saved.
        device : str
            'cuda' or 'cpu'
        """

        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = get_model(name="htdemucs")
        self.model.to(device)
        self.model.eval()

        self.sources = self.model.sources

    def separate_file(self, audio_path):
        """Run separation on a single file."""


        wav, sr = sf.read(audio_path)
        wav = torch.tensor(wav).float()
        # convert to channels x samples
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.T
        wav = wav.to(self.device)

        with torch.no_grad():
            sources = apply_model(self.model, wav[None])[0]

        track_name = Path(audio_path).stem
        track_dir = self.output_dir / track_name
        track_dir.mkdir(exist_ok=True)


        vocals_stem = sources[0].cpu()
        instrumental = sources[1] + sources[2] + sources[3]
        instrumental_stem = instrumental.cpu()

        sf.write(track_dir / f"voice.wav",
                vocals_stem.T.numpy(),
                sr)
        
        sf.write(track_dir / f"instrumental.wav",
                instrumental_stem.T.numpy(),
                sr)
        


    def separate_from_dataframe(self, df, path_column):
        """
        Run separation over all files in a dataframe.

        Args
        ----
        df : pandas.DataFrame
        path_column : str
            Column containing audio file paths
        """

        for audio_path in df[path_column]:
            start_time = time.time()
            print(f"Processing {audio_path}")
            self.separate_file(audio_path)
            print(f"File processed in {time.time()-start_time} seconds.")
