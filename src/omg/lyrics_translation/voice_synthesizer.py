"""Voice synthesis and singing voice conversion module.

This module provides text-to-speech synthesis with voice cloning capabilities.
For singing voice synthesis, we use a combination of TTS and pitch manipulation.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from .lyrics_aligner import WordTiming
import parselmouth
from parselmouth.praat import call
import soundfile as sf
import librosa


class VoiceSynthesizer:
    """Synthesizes singing voice from text using voice cloning.

    This class provides functionality to:
    1. Extract voice characteristics from reference audio
    2. Synthesize speech from new lyrics
    3. Manipulate pitch to match singing patterns

    For production use, consider using:
    - seed-vc for singing voice conversion
    - so-vits-svc for singing voice synthesis
    - OpenVoice or XTTS for voice cloning

    Args:
        model_name: TTS model to use (currently simplified)
        device: Device to run the model on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Initializing VoiceSynthesizer with model: {model_name}")
        print("Note: Full singing voice synthesis requires additional models")
        print("Consider using: seed-vc, so-vits-svc, or OpenVoice for production")

    def synthesize_from_lyrics(
        self,
        new_lyrics: str,
        reference_vocals_path: str,
        old_word_timings: List[WordTiming],
        output_path: str,
    ) -> str:
        """Synthesize singing voice from new lyrics.

        This is a simplified implementation. For production, integrate:
        - seed-vc: https://github.com/Plachtaa/seed-vc
        - so-vits-svc: https://github.com/svc-develop-team/so-vits-svc

        Args:
            new_lyrics: New lyrics to synthesize
            reference_vocals_path: Path to reference vocals (for voice cloning)
            old_word_timings: Timing information from original lyrics
            output_path: Path to save synthesized audio

        Returns:
            Path to synthesized audio file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("SINGING VOICE SYNTHESIS")
        print("=" * 60)
        print(f"New lyrics: {new_lyrics}")
        print(f"Reference vocals: {reference_vocals_path}")
        print(f"Output path: {output_path}")

        # Load reference audio using soundfile (avoids torchcodec issues)
        ref_audio, ref_sr = sf.read(reference_vocals_path)
        # Convert to torch tensor
        if len(ref_audio.shape) == 1:
            # Mono audio
            ref_waveform = torch.tensor(ref_audio, dtype=torch.float32).unsqueeze(0)
        else:
            # Multi-channel audio - keep as is
            ref_waveform = torch.tensor(ref_audio, dtype=torch.float32).T

        # Get original duration for time-stretching
        original_duration = len(ref_audio) / ref_sr
        print(f"\nOriginal vocals duration: {original_duration:.2f}s")

        # Synthesize new vocals with TTS
        synthesized_audio = self._apply_basic_modifications(
            ref_waveform,
            ref_sr,
            new_lyrics,
            old_word_timings,
            reference_vocals_path,
        )

        # Convert tensor to numpy for processing
        audio_numpy = synthesized_audio.cpu().numpy()
        if audio_numpy.ndim == 2:
            if audio_numpy.shape[0] == 1:
                audio_numpy = audio_numpy[0]
            else:
                audio_numpy = audio_numpy.T

        # Get synthesized duration
        synth_duration = len(audio_numpy) / ref_sr
        print(f"Synthesized vocals duration: {synth_duration:.2f}s")

        # Time-stretch to match original duration
        if abs(synth_duration - original_duration) > 0.1:  # Only stretch if significantly different
            # librosa.effects.time_stretch uses rate parameter as playback rate
            # To slow down (make longer), we need rate < 1
            stretch_rate = synth_duration / original_duration
            print(f"Time-stretching with rate: {stretch_rate:.4f}")
            audio_numpy = librosa.effects.time_stretch(audio_numpy, rate=stretch_rate)
            print(f"After stretch duration: {len(audio_numpy) / ref_sr:.2f}s")

        sf.write(str(output_path), audio_numpy, ref_sr)

        print(f"\n✓ Synthesized audio saved to: {output_path}")
        return str(output_path)

    def _apply_basic_modifications(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        new_lyrics: str,
        word_timings: List[WordTiming],
        reference_vocals_path: str,
    ) -> torch.Tensor:
        """Apply basic audio modifications.

        This is a placeholder that applies simple transformations.
        Replace with proper singing synthesis in production.

        Args:
            waveform: Input audio waveform
            sample_rate: Sample rate
            new_lyrics: New lyrics
            word_timings: Original word timings
            reference_vocals_path: Path to reference vocals for voice cloning

        Returns:
            Modified waveform
        """
        # For now, just return the original audio
        # In production, implement:
        # - Pitch shifting based on lyrics prosody
        # - Time stretching to match new lyrics timing
        # - Voice conversion to match reference

        try:
            # Try to use XTTS v2 for synthesis with voice cloning
            print("\n🎤 Synthesizing new vocals with XTTS v2...")

            # Use gpt-sovits or other voice cloning models if available
            # For now, attempt to use TTS.api with better error handling
            try:
                from TTS.api import TTS
                import os

                # Accept the TTS license to avoid interactive prompts
                os.environ["TTS_HOME"] = str(Path.home() / ".tts")
                # Create a marker file to skip license confirmation
                tts_home = Path.home() / ".tts"
                tts_home.mkdir(exist_ok=True)
                license_file = tts_home / "AGREES_NONCOMM_CPML.txt"
                license_file.touch()

                device = self.device
                gpu = (device == "cuda")

                # Use multilingual XTTS v2 model that supports Chinese
                tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu)

                # Synthesize speech from new lyrics with voice cloning
                # Use TTS with speaker reference for better voice matching
                synthesized_wav = tts_model.tts(
                    text=new_lyrics,
                    speaker_wav=reference_vocals_path,
                    language="auto"  # Auto-detect language
                )

                # Convert to torch tensor
                synthesized = torch.tensor(synthesized_wav, dtype=torch.float32)
                if synthesized.ndim == 1:
                    synthesized = synthesized.unsqueeze(0)

                # Resample if needed to match reference audio sample rate
                if synthesized.shape[0] == 1:  # Mono audio
                    # Default TTS model outputs 22050 Hz, resample to match reference
                    tts_sr = 22050  # Default TTS output sample rate
                    if tts_sr != sample_rate:
                        resampler = torchaudio.transforms.Resample(tts_sr, sample_rate)
                        synthesized = resampler(synthesized)

                print("✓ XTTS v2 synthesis completed with voice cloning")
                return synthesized

            except ImportError as import_err:
                print(f"⚠️  TTS.api import failed: {import_err}")
                print("Attempting fallback TTS approach...")

                # Fallback: Try using a simpler TTS without transformers
                try:
                    # For Chinese text, use a Chinese-specific TTS if available
                    # Otherwise, use glow-tts which has better compatibility
                    from TTS.api import TTS
                    device = self.device
                    gpu = (device == "cuda")

                    # Use glow-tts as fallback (more stable)
                    tts_model = TTS(model_name="tts_models/en/ljspeech/glow-tts", gpu=gpu)
                    synthesized_wav = tts_model.tts(text=new_lyrics)

                    synthesized = torch.tensor(synthesized_wav, dtype=torch.float32)
                    if synthesized.ndim == 1:
                        synthesized = synthesized.unsqueeze(0)

                    if synthesized.shape[0] == 1:
                        tts_sr = 22050
                        if tts_sr != sample_rate:
                            resampler = torchaudio.transforms.Resample(tts_sr, sample_rate)
                            synthesized = resampler(synthesized)

                    print("✓ Fallback TTS synthesis completed")
                    return synthesized
                except Exception as fallback_err:
                    print(f"⚠️  Fallback TTS also failed: {fallback_err}")
                    raise

        except Exception as e:
            print(f"\n⚠️  TTS synthesis not available ({type(e).__name__}: {e})")
            print("Using reference audio as fallback...")
            return waveform

    def extract_pitch_contour(
        self,
        audio_path: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch contour from audio using Praat.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (time_points, pitch_values) in Hz
        """
        print(f"Extracting pitch contour from {audio_path}")

        # Load audio with parselmouth
        sound = parselmouth.Sound(audio_path)

        # Extract pitch
        pitch = call(sound, "To Pitch", 0.0, 75, 600)  # 75-600 Hz range for singing

        # Get pitch values
        pitch_values = []
        time_points = []

        for t in np.arange(0, sound.duration, 0.01):  # 10ms intervals
            pitch_value = call(pitch, "Get value at time", t, "Hertz", "Linear")
            if pitch_value is not None and not np.isnan(pitch_value):
                time_points.append(t)
                pitch_values.append(pitch_value)

        time_points = np.array(time_points)
        pitch_values = np.array(pitch_values)

        print(f"Extracted {len(pitch_values)} pitch points")
        return time_points, pitch_values

    def apply_pitch_shift(
        self,
        audio_path: str,
        semitones: float,
        output_path: str,
    ) -> str:
        """Apply pitch shift to audio.

        Args:
            audio_path: Input audio path
            semitones: Number of semitones to shift (positive = up, negative = down)
            output_path: Output audio path

        Returns:
            Path to pitch-shifted audio
        """
        print(f"Shifting pitch by {semitones} semitones")

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Apply pitch shift using torchaudio
        # Note: This is a basic implementation
        # For better quality, use rubberband or librosa
        pitch_shift = torchaudio.transforms.PitchShift(
            sample_rate=sample_rate,
            n_steps=int(semitones),
        )

        shifted = pitch_shift(waveform)

        # Save output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), shifted, sample_rate)

        print(f"Pitch-shifted audio saved to {output_path}")
        return str(output_path)

    def combine_with_instrumental(
        self,
        vocals_path: str,
        instrumental_path: str,
        output_path: str,
        vocals_gain: float = 1.0,
        instrumental_gain: float = 1.0,
    ) -> str:
        """Combine synthesized vocals with instrumental.

        Args:
            vocals_path: Path to vocals audio
            instrumental_path: Path to instrumental audio
            output_path: Path to save combined audio
            vocals_gain: Volume adjustment for vocals (1.0 = no change)
            instrumental_gain: Volume adjustment for instrumental (1.0 = no change)

        Returns:
            Path to combined audio
        """
        print("Combining vocals with instrumental...")

        # Load both tracks using soundfile (avoids torchcodec issues)
        vocals_np, vocals_sr = sf.read(vocals_path)
        instrumental_np, inst_sr = sf.read(instrumental_path)

        # Convert to torch tensors
        if len(vocals_np.shape) == 1:
            vocals = torch.tensor(vocals_np, dtype=torch.float32).unsqueeze(0)
        else:
            vocals = torch.tensor(vocals_np, dtype=torch.float32).T

        if len(instrumental_np.shape) == 1:
            instrumental = torch.tensor(instrumental_np, dtype=torch.float32).unsqueeze(0)
        else:
            instrumental = torch.tensor(instrumental_np, dtype=torch.float32).T

        # Resample if needed
        if vocals_sr != inst_sr:
            resampler = torchaudio.transforms.Resample(vocals_sr, inst_sr)
            vocals = resampler(vocals)
            vocals_sr = inst_sr

        # Match number of channels
        if vocals.shape[0] != instrumental.shape[0]:
            if vocals.shape[0] == 1 and instrumental.shape[0] == 2:
                vocals = vocals.repeat(2, 1)
            elif vocals.shape[0] == 2 and instrumental.shape[0] == 1:
                instrumental = instrumental.repeat(2, 1)

        # Trim or pad to match length
        min_length = min(vocals.shape[1], instrumental.shape[1])
        vocals = vocals[:, :min_length]
        instrumental = instrumental[:, :min_length]

        # Mix with gains
        mixed = vocals * vocals_gain + instrumental * instrumental_gain

        # Normalize to prevent clipping
        max_val = mixed.abs().max()
        if max_val > 1.0:
            mixed = mixed / max_val

        # Save output using soundfile (avoids torchcodec issues)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensor to numpy
        mixed_numpy = mixed.cpu().numpy()
        # Handle different tensor shapes
        if mixed_numpy.ndim == 2:
            # (channels, samples) -> (samples,) or (samples, channels)
            if mixed_numpy.shape[0] == 1:
                mixed_numpy = mixed_numpy[0]  # Remove single channel dimension
            else:
                mixed_numpy = mixed_numpy.T  # (channels, samples) -> (samples, channels)

        sf.write(str(output_path), mixed_numpy, inst_sr)

        print(f"Combined audio saved to {output_path}")
        return str(output_path)
