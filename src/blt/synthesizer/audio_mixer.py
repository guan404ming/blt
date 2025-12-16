"""Audio mixing module for combining multiple audio tracks."""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Union
from scipy import signal


class AudioMixer:
    """Mixes multiple audio tracks together.

    Combines two audio files by overlapping them, automatically handling
    different sample rates and lengths.

    Args:
        normalize: Whether to normalize output to prevent clipping (default: True)
        output_format: Output file format (default: 'wav')
    """

    def __init__(
        self,
        normalize: bool = True,
        output_format: str = "wav",
    ):
        self.normalize = normalize
        self.output_format = output_format

    def _resample_audio(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio to a target sample rate.

        Args:
            audio: Audio data (samples x channels or just samples for mono)
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio data
        """
        if orig_sr == target_sr:
            return audio

        # Calculate resampling ratio
        ratio = target_sr / orig_sr

        # Handle mono and stereo
        if audio.ndim == 1:
            # Mono audio
            num_samples = int(len(audio) * ratio)
            resampled = signal.resample(audio, num_samples)
        else:
            # Stereo/multi-channel audio - resample each channel
            num_samples = int(audio.shape[0] * ratio)
            resampled = np.zeros((num_samples, audio.shape[1]))
            for ch in range(audio.shape[1]):
                resampled[:, ch] = signal.resample(audio[:, ch], num_samples)

        return resampled

    def mix(
        self,
        audio1_path: Union[str, Path],
        audio2_path: Union[str, Path],
        output_path: Union[str, Path],
        volume1: float = 1.0,
        volume2: float = 1.0,
    ) -> str:
        """Mix two audio files together.

        Args:
            audio1_path: Path to first audio file
            audio2_path: Path to second audio file
            output_path: Path for output mixed audio file
            volume1: Volume multiplier for first audio (0.0 to 1.0+)
            volume2: Volume multiplier for second audio (0.0 to 1.0+)

        Returns:
            Path to the output mixed audio file
        """
        audio1_path = Path(audio1_path)
        audio2_path = Path(audio2_path)
        output_path = Path(output_path)

        if not audio1_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio1_path}")
        if not audio2_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio2_path}")

        print("Mixing audio files...")
        print(f"  Track 1: {audio1_path} (volume: {volume1})")
        print(f"  Track 2: {audio2_path} (volume: {volume2})")

        # Read audio files
        audio1, sr1 = sf.read(str(audio1_path))
        audio2, sr2 = sf.read(str(audio2_path))

        # Resample if sample rates don't match
        if sr1 != sr2:
            print(
                f"  Sample rates differ ({sr1} Hz vs {sr2} Hz), "
                f"resampling to {sr1} Hz..."
            )
            audio2 = self._resample_audio(audio2, sr2, sr1)
        sample_rate = sr1

        # Convert to mono if stereo (or handle stereo mixing)
        # For simplicity, we'll handle both mono and stereo
        if audio1.ndim == 1:
            audio1 = audio1.reshape(-1, 1)
        if audio2.ndim == 1:
            audio2 = audio2.reshape(-1, 1)

        # Make both audios have the same number of channels
        if audio1.shape[1] != audio2.shape[1]:
            # Convert both to stereo if one is stereo and one is mono
            max_channels = max(audio1.shape[1], audio2.shape[1])
            if audio1.shape[1] < max_channels:
                audio1 = np.repeat(audio1, max_channels, axis=1)
            if audio2.shape[1] < max_channels:
                audio2 = np.repeat(audio2, max_channels, axis=1)

        # Apply volume adjustments
        audio1 = audio1 * volume1
        audio2 = audio2 * volume2

        # Pad shorter audio to match lengths
        max_length = max(len(audio1), len(audio2))
        if len(audio1) < max_length:
            padding = np.zeros((max_length - len(audio1), audio1.shape[1]))
            audio1 = np.vstack([audio1, padding])
        if len(audio2) < max_length:
            padding = np.zeros((max_length - len(audio2), audio2.shape[1]))
            audio2 = np.vstack([audio2, padding])

        # Mix the audio by adding them together
        mixed = audio1 + audio2

        # Normalize to prevent clipping
        if self.normalize:
            max_val = np.abs(mixed).max()
            if max_val > 1.0:
                mixed = mixed / max_val
                print(f"  Normalized by factor of {max_val:.2f} to prevent clipping")

        # Convert back to mono if it was originally mono
        if mixed.shape[1] == 1:
            mixed = mixed.reshape(-1)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save mixed audio
        sf.write(str(output_path), mixed, sample_rate)

        print("✓ Mixing complete!")
        print(f"  Output: {output_path}")

        return str(output_path)
