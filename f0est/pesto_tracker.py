import torch
import numpy as np
import soundfile as sf
import pesto
from pathlib import Path
from typing import Tuple, Optional
from scipy.signal import resample_poly

class PESTOTracker:
    """
    Pitch detection using the PESTO (Probabilistic Expectation-Maximization Source Tracking Outline) algorithm.
    
    PESTO is a PyTorch-based pitch detection model that works best with 16kHz audio.
    It provides both pitch estimates and confidence scores for each frame.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        step_size: float = 0.01,
        sample_rate: int = 16000,
        use_gpu: bool = True
    ):
        """
        Initialize the PESTO tracker.
        
        Args:
            confidence_threshold (float): Confidence threshold for filtering pitch estimates (0-1).
                                        Estimates below this threshold are considered unvoiced.
                                        Default: 0.5
            step_size (float): Hop length in seconds for pitch estimation. Default: 0.01 (10ms)
            sample_rate (int): Target sample rate for audio processing. PESTO works best at 16kHz.
                              Default: 16000
            use_gpu (bool): Whether to use GPU acceleration if available. Default: True
        """
        self.confidence_threshold = confidence_threshold
        self.step_size = step_size #convert seconds to miliseconds
        self.sample_rate = sample_rate
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
    
    def extract_f0(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch (f0) from an audio file using PESTO.
        
        Args:
            audio_path (str): Path to the audio file.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - pitches: Detected pitch values in Hz (unvoiced frames set to 0)
                - timesteps: Time positions for each frame in seconds
                - voicing_confidence: Confidence scores for each frame
        
        Raises:
            FileNotFoundError: If the audio file does not exist.
            RuntimeError: If PESTO model fails to process the audio.
        """
        # Validate file path
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio using soundfile
            wav, sr = sf.read(str(audio_path))
            
            # Validate audio
            if len(wav) == 0:
                raise ValueError("Audio file is empty or could not be loaded.")
            
            # Resample if necessary (PESTO works best with 16kHz)
            if sr != self.sample_rate:
                wav = resample_poly(wav, self.sample_rate, sr)
                sr = self.sample_rate
            elif sr != self.sample_rate:
                raise ValueError(
                    f"Audio sample rate {sr} does not match target {self.sample_rate}. "
                    "Install librosa for automatic resampling."
                )
            
            # Convert to torch tensor
            wav = torch.tensor(wav, dtype=torch.float32)
            
            # Handle multi-channel audio
            if wav.ndim == 1:
                # Already mono
                pass
            else:
                # Multi-channel: transpose to channels x samples then take mean
                wav = wav.T
                wav = wav.mean(dim=0)  # Average across channels to get mono
            
            # Move to appropriate device
            if self.use_gpu:
                wav = wav.to(self.device)
            
            # Run PESTO pitch detection
            # Returns: timesteps, pitch, confidence, activations
            #timesteps, pitch, confidence, _ = pesto.predict(wav, sr, step_size=self.step_size*1000)# convert seconds to miliseconds
            chunk_sec = 15
            chunk_samples = int(chunk_sec * self.sample_rate)

            pitch = []
            confidence = []
            timesteps = []

            for start in range(0, len(wav), chunk_samples):
                chunk = wav[start:start + chunk_samples]

                chunk_timesteps, chunk_pitch, chunk_confidence, _ = pesto.predict(
                    chunk, self.sample_rate, step_size=self.step_size * 1000
                )

                # Avoid in-place ops on inference tensors returned by PyTorch models
                chunk_timesteps=chunk_timesteps/1000 # conver to seconds
                chunk_timesteps = chunk_timesteps + (start / self.sample_rate) #convert start/sr inito miliseconds
                
                # Convert chunk outputs to numpy and aggregate
                if isinstance(chunk_pitch, torch.Tensor):
                    chunk_pitch = chunk_pitch.detach().cpu().numpy()
                else:
                    chunk_pitch = np.asarray(chunk_pitch)

                if isinstance(chunk_confidence, torch.Tensor):
                    chunk_confidence = chunk_confidence.detach().cpu().numpy()
                else:
                    chunk_confidence = np.asarray(chunk_confidence)

                if isinstance(chunk_timesteps, torch.Tensor):
                    chunk_timesteps = chunk_timesteps.detach().cpu().numpy()
                else:
                    chunk_timesteps = np.asarray(chunk_timesteps)

                pitch.append(chunk_pitch)
                confidence.append(chunk_confidence)
                timesteps.append(chunk_timesteps)

    
            # Concatenate chunk outputs
            timesteps = np.concatenate(timesteps) if timesteps else np.array([], dtype=np.float32)
            pitch = np.concatenate(pitch) if pitch else np.array([], dtype=np.float32)
            confidence = np.concatenate(confidence) if confidence else np.array([], dtype=np.float32)
            
            # Apply confidence filtering: set unvoiced frames (low confidence) to 0 Hz
            final_pitches = np.where(
                confidence > self.confidence_threshold,
                pitch,
                0  # 0 Hz represents unvoiced
            )
            
            return final_pitches, timesteps, confidence
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract pitch from {audio_path}: {str(e)}")
    
    def extract_f0_with_voicing(
        self,
        audio_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch (f0) from an audio file with separate voicing information.
        
        Args:
            audio_path (str): Path to the audio file.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - pitches: Detected pitch values in Hz (only voiced frames)
                - times: Time positions for each frame in seconds
                - voicing: Binary voicing labels (1 = voiced, 0 = unvoiced)
                - confidence: Confidence scores for each frame
        """
        pitches, timesteps, confidence = self.extract_f0(audio_path)
        
        # Create voicing vector
        voicing = (confidence > self.confidence_threshold).astype(int)
        
        # Extract only voiced pitch values
        voiced_pitches = pitches[voicing == 1]
        
        return voiced_pitches, timesteps, voicing, confidence
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update the confidence threshold for voicing detection.
        
        Args:
            threshold (float): New confidence threshold (0-1).
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        self.confidence_threshold = threshold

    def save_pitch_contour(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        include_confidence: bool = True
    ) -> Path:
        """
        Extract pitch from an audio file and save its contour to a CSV file.

        Args:
            audio_path (str): Path to the input audio file.
            output_path (Optional[str]): Path to the output CSV file.
                                        If None, saves next to the input audio as "<stem>.f0.csv".
            include_confidence (bool): Whether to include confidence and voicing columns.

        Returns:
            Path: Path to the saved CSV file.
        """
        pitches, timesteps, confidence = self.extract_f0(audio_path)

        audio_path = Path(audio_path)
        if output_path is None:
            output_path = audio_path.with_suffix(".f0.csv")
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if include_confidence:
            voicing = (confidence > self.confidence_threshold).astype(int)
            data = np.column_stack((timesteps/1000, pitches, confidence, voicing))
            header = "time_sec,f0_hz,confidence,voicing"
        else:
            data = np.column_stack((timesteps, pitches))
            header = "time_sec,f0_hz"

        np.savetxt(
            output_path,
            data,
            delimiter=",",
            header=header,
            comments="",
            fmt="%.8f"
        )

        return output_path

