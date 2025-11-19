"""Melody encoder implementation using librosa."""

import torch
import librosa
import numpy as np
import pretty_midi

# I will not inherit from BaseAudioEncoder as it's not a perfect fit,
# but rather implement the encode_to_string method directly.


class MelodyEncoder:
    """Melody encoder for extracting a melodic line from audio and representing it as a string."""

    def __init__(self):
        """Initialize the MelodyEncoder."""
        pass  # No specific setup needed for librosa

    def encode_to_string(self, audio: torch.Tensor, sample_rate: int) -> str:
        """Extract a melodic line from audio and represent it as a string.

        Args:
            audio: Audio waveform tensor [channels, samples] or [batch, channels, samples]
            sample_rate: Sample rate of the input audio

        Returns:
            A string representing the extracted melody (e.g., "p60:d0.5 p62:d0.5").
        """
        # Ensure audio is on CPU and in numpy format for librosa
        audio_np = audio.cpu().numpy()

        # Ensure audio_np is mono and 1D
        if audio_np.ndim > 1:
            audio_np = librosa.to_mono(audio_np)

        # Correcting pitch extraction to use librosa.pyin for fundamental frequency (f0)
        # Using a default frame_length and hop_length.
        frame_length = 2048
        hop_length = 512
        f0, voiced_flag, _ = librosa.pyin(
            y=audio_np,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )

        # Convert f0 to MIDI notes
        midi_notes = librosa.hz_to_midi(f0)

        # Process midi_notes to get a sequence of pitch:duration
        melody_events = []
        current_note = None
        current_duration = 0.0

        # Iterate over the estimated pitches
        # The duration of one frame is hop_length / sample_rate
        frame_time = hop_length / sample_rate

        for i, (pitch, voiced) in enumerate(zip(midi_notes, voiced_flag)):
            if voiced and not np.isnan(pitch):
                # Quantize pitch to nearest integer MIDI note
                quantized_pitch = int(np.round(pitch))

                if current_note == quantized_pitch:
                    current_duration += frame_time
                else:
                    if current_note is not None:
                        # Append the previous note event if it existed
                        melody_events.append(f"p{current_note}:d{current_duration:.2f}")
                    # Start a new note
                    current_note = quantized_pitch
                    current_duration = frame_time
            else:
                # If not voiced or NaN, treat as a break or end of note
                if current_note is not None:
                    # Append the previous note event before a break
                    melody_events.append(f"p{current_note}:d{current_duration:.2f}")
                current_note = None
                current_duration = 0.0

        # Add the last note if any
        if current_note is not None:
            melody_events.append(f"p{current_note}:d{current_duration:.2f}")

        melody_str = " ".join(melody_events)
        return melody_str

    def get_sample_rate(self) -> int:
        """MelodyEncoder does not have a fixed sample rate; it adapts to input."""
        return 0  # Or raise NotImplementedError

    def get_num_codebooks(self) -> int:
        """MelodyEncoder does not use codebooks."""
        return 0  # Or raise NotImplementedError

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """MelodyEncoder does not support decoding."""
        raise NotImplementedError("MelodyEncoder does not support decoding.")
