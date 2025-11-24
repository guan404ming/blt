"Lyrics alignment module using forced alignment."

import torch
import torchaudio
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import soundfile as sf
from pypinyin import pinyin, Style
from ctc_forced_aligner import (
    AlignmentSingleton,
    generate_emissions,
    get_alignments,
    get_spans,
)


@dataclass
class WordTiming:
    """Timing information for a word in the lyrics."""

    word: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    score: float  # Alignment confidence score


class LyricsAligner:
    """Aligns lyrics text with audio using forced alignment.

    This class uses CTC-based forced alignment with wav2vec2/MMS models
    to align lyrics words with their timing in the audio.

    Args:
        model_name: HuggingFace model for forced alignment.
                   Default: "MahmoudAshraf/mms-300m-1130-forced-aligner"
        device: Device to run the model on ('cuda' or 'cpu')
        language: Language code (e.g., 'eng', 'spa', 'fra', 'zho')
    """

    def __init__(self, model_name: str = None, device: str = None):
        print("Loading alignment model using ctc_forced_aligner.AlignmentSingleton")
        aligner_instance = AlignmentSingleton()
        self.model = aligner_instance.alignment_model
        self.tokenizer = aligner_instance.alignment_tokenizer
        self.device = device

    def align(
        self,
        audio_path: str,
        lyrics: str,
    ) -> List[WordTiming]:
        """Align lyrics with audio to get word-level timing.

        Args:
            audio_path: Path to audio file (preferably vocals-only)
            lyrics: Full lyrics text

        Returns:
            List of WordTiming objects with timing for each word
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio using soundfile instead of torchaudio (avoids torchcodec issues)
        print(f"Loading audio from {audio_path}")
        audio_data, sample_rate = sf.read(str(audio_path))

        # Convert to torch tensor
        if len(audio_data.shape) == 1:
            # Mono audio
            waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
        else:
            # Stereo or multi-channel - convert to mono
            waveform = torch.tensor(audio_data, dtype=torch.float32).mean(dim=1, keepdim=True).T

        # Resample to 16kHz if needed (required by most alignment models)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Prepare text (split into words)
        words = lyrics.strip().split()
        print(f"Aligning {len(words)} words with audio...")

        # Generate emissions
        emissions, stride = generate_emissions(
            self.model,
            waveform.squeeze(),
            batch_size=4,
        )

        # Convert Chinese lyrics to Pinyin for alignment (tokenizer only supports Latin characters)
        # Get pinyin representation of the lyrics
        pinyin_text = ""
        for char in lyrics:
            if '\u4e00' <= char <= '\u9fff':  # Check if Chinese character
                py = pinyin(char, style=Style.NORMAL, heteronym=False)
                if py and py[0]:
                    pinyin_text += py[0][0].lower() + " "
            elif char.isalpha():
                # Keep alphabetic characters
                pinyin_text += char.lower() + " "
            elif char.isspace():
                # Keep spaces to separate words
                pinyin_text += " "
            # Skip all other characters (punctuation, numbers, special chars like "-")

        # Split pinyin into tokens
        tokens_list = pinyin_text.split()
        tokens_list = [t for t in tokens_list if t]  # Remove empty strings

        try:
            segments, scores, blank_id = get_alignments(
                emissions,
                tokens_list,
                self.tokenizer,
            )

            # Get word-level spans
            spans = get_spans(tokens_list, segments, blank_id)
        except (AssertionError, ValueError) as e:
            # If alignment fails, use a simple approximation by splitting the audio evenly
            print(f"Warning: Alignment failed ({e}). Using uniform time distribution.")
            # Get the total audio duration
            total_frames = emissions.shape[0]
            stride = total_frames / len(words) if len(words) > 0 else total_frames
            spans = [(int(i * stride), int((i + 1) * stride)) for i in range(len(words))]
            scores = []

        # Convert to WordTiming objects
        word_timings = []
        for span, word in zip(spans, words):
            # span is (start_frame, end_frame)
            # Convert frames to seconds
            start_sec = span[0] * stride / sample_rate
            end_sec = span[1] * stride / sample_rate

            # Calculate average score for this word
            word_score = scores[span[0] : span[1]].mean().item() if len(scores) > 0 else 1.0

            word_timings.append(
                WordTiming(
                    word=word,
                    start=start_sec,
                    end=end_sec,
                    score=word_score,
                )
            )

        print(f"Alignment complete! Aligned {len(word_timings)} words.")
        return word_timings

    def get_alignment_dict(self, word_timings: List[WordTiming]) -> Dict[str, List[float]]:
        """Convert word timings to a dictionary format.

        Args:
            word_timings: List of WordTiming objects

        Returns:
            Dictionary mapping words to [start, end] times
        """
        return {
            wt.word: [wt.start, wt.end] for wt in word_timings
        }

    def save_alignment(
        self,
        word_timings: List[WordTiming],
        output_path: str,
    ):
        """Save word alignments to a text file.

        Args:
            word_timings: List of WordTiming objects
            output_path: Path to save alignment file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Word\tStart\tEnd\tScore\n")
            for wt in word_timings:
                f.write(f"{wt.word}\t{wt.start:.3f}\t{wt.end:.3f}\t{wt.score:.3f}\n")

        print(f"Alignment saved to {output_path}")

    def print_alignment(self, word_timings: List[WordTiming], max_words: int = 10):
        """Print word alignments to console.

        Args:
            word_timings: List of WordTiming objects
            max_words: Maximum number of words to print (0 for all)
        """
        print("\nWord Alignment:")
        print("=" * 60)
        print(f"{'Word':<20} {'Start':>10} {'End':>10} {'Score':>10}")
        print("-" * 60)

        words_to_show = word_timings if max_words == 0 else word_timings[:max_words]
        for wt in words_to_show:
            print(f"{wt.word:<20} {wt.start:>10.3f} {wt.end:>10.3f} {wt.score:>10.3f}")

        if max_words > 0 and len(word_timings) > max_words:
            print(f"... ({len(word_timings) - max_words} more words)")

        print("=" * 60)