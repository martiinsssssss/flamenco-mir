import os
import time
import torch
from pathlib import Path
import soundfile as sf

from demucs.pretrained import get_model
from demucs.apply import apply_model

from spleeter.separator import Separator


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


        vocals_stem = sources[3].cpu()
        instrumental = sources[0] + sources[1] + sources[2]
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




class SpleeterSeparator:
    def __init__(self, output_dir="separated", stems=2):
        """
        Simple Spleeter wrapper for batch source separation.

        Args
        ----
        output_dir : str
            Where separated stems will be saved.
        stems : int
            Number of stems (2, 4, or 5)
            - 2: vocals and accompaniment
            - 4: vocals, drums, bass, other
            - 5: vocals, drums, bass, piano, other
        """

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the separator with the specified stem configuration
        self.stems = stems
        self.separator = Separator(f'spleeter:{stems}stems')

    def separate_file(self, audio_path):
        """Run separation on a single file."""

        audio_path = Path(audio_path)
        track_name = audio_path.stem
        track_dir = self.output_dir / track_name
        track_dir.mkdir(exist_ok=True)

        # Spleeter expects the output directory and creates subdirectories
        # It will create track_name/vocals.wav, track_name/accompaniment.wav, etc.
        self.separator.separate_to_file(
            str(audio_path),
            str(self.output_dir),
            codec='wav'
        )

        # For 2-stem separation, create an instrumental file
        if self.stems == 2:
            # Spleeter creates: output_dir/track_name/vocals.wav and accompaniment.wav
            # Rename accompaniment to instrumental for consistency
            src_accompaniment = track_dir / "accompaniment.wav"
            dst_instrumental = track_dir / "instrumental.wav"
            if src_accompaniment.exists() and not dst_instrumental.exists():
                src_accompaniment.rename(dst_instrumental)

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
