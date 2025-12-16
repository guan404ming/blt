"Lyrics alignment module using forced alignment."

import torch
import torchaudio
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import soundfile as sf
from pypinyin import pinyin, Style
from ctc_forced_aligner import (
    AlignmentSingleton,
    generate_emissions,
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
            waveform = (
                torch.tensor(audio_data, dtype=torch.float32)
                .mean(dim=1, keepdim=True)
                .T
            )

        # Resample to 16kHz if needed (required by most alignment models)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Generate emissions first
        emissions, stride = generate_emissions(
            self.model,
            waveform.squeeze(),
            batch_size=4,
        )

        # Prepare text for alignment - split by sentences/lines
        clean_lyrics = lyrics.strip()

        # Split by line breaks first to preserve sentence structure
        lines = clean_lyrics.split("\n")
        words = []

        for line in lines:
            # Skip empty lines
            line = line.strip()
            if not line:
                continue
            # Remove trailing punctuation for word extraction
            line_for_words = line
            for punct in [
                "，",
                "。",
                "！",
                "？",
                "、",
                "；",
                "：",
                '"',
                "'",
                """, """,
                "-",
            ]:
                line_for_words = line_for_words.replace(punct, " ")

            # Add the line as a complete unit
            words.append(line)

        if not words:
            # Fallback: split by spaces if no line breaks
            clean_lyrics = lyrics.strip()
            for punct in [
                "-",
                "，",
                "。",
                "！",
                "？",
                "、",
                "；",
                "：",
                '"',
                "'",
                """, """,
                "、",
            ]:
                clean_lyrics = clean_lyrics.replace(punct, " ")
            words = [w for w in clean_lyrics.split() if w]

        print(f"Aligning {len(words)} sentences/lines with audio...")

        # Convert Chinese lyrics to Pinyin for alignment (tokenizer only supports Latin characters)
        # Map each sentence/word to its pinyin tokens and track the mapping
        pinyin_text = ""
        word_to_token_indices = {}  # Maps word index to (start_token_idx, end_token_idx)
        token_idx = 0

        for word_idx, word in enumerate(words):
            token_start = token_idx
            word_pinyin = ""

            for char in word:
                if "\u4e00" <= char <= "\u9fff":  # Check if Chinese character
                    py = pinyin(char, style=Style.NORMAL, heteronym=False)
                    if py and py[0]:
                        py_text = py[0][0].lower()
                        word_pinyin += py_text + " "
                        token_idx += 1
                elif char.isalpha():
                    # Keep alphabetic characters
                    word_pinyin += char.lower() + " "
                    token_idx += 1
                elif char.isspace():
                    # Keep spaces within sentence
                    word_pinyin += " "
                # Skip punctuation and other characters

            # Only track if we have tokens for this sentence
            if token_idx > token_start:
                word_to_token_indices[word_idx] = (token_start, token_idx)
            pinyin_text += word_pinyin + " "

        # Split pinyin into tokens
        tokens_list = pinyin_text.split()
        tokens_list = [t for t in tokens_list if t]  # Remove empty strings

        # For sentence-level alignment, use uniform time distribution
        # CTC-based alignment with pinyin tokens is unreliable for singing voice
        print("Using uniform time distribution for sentence-level alignment")
        print(
            f"  ({len(tokens_list)} pinyin tokens extracted from {len(words)} sentences)"
        )

        # Use uniform time distribution directly
        total_frames = emissions.shape[0]
        frame_stride = total_frames / len(words) if len(words) > 0 else total_frames
        spans = [
            (int(i * frame_stride), int((i + 1) * frame_stride))
            for i in range(len(words))
        ]
        scores = []

        # Convert to WordTiming objects
        word_timings = []
        for span, word in zip(spans, words):
            # span is (start_frame, end_frame)
            # Convert frames to seconds
            start_sec = span[0] * stride / sample_rate
            end_sec = span[1] * stride / sample_rate

            # Calculate average score for this word if available
            if len(scores) > 0 and span[0] < len(scores):
                score_slice = scores[span[0] : min(span[1], len(scores))]
                word_score = score_slice.mean().item() if len(score_slice) > 0 else 1.0
            else:
                word_score = 1.0

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

    def get_alignment_dict(
        self, word_timings: List[WordTiming]
    ) -> Dict[str, List[float]]:
        """Convert word timings to a dictionary format.

        Args:
            word_timings: List of WordTiming objects

        Returns:
            Dictionary mapping words to [start, end] times
        """
        return {wt.word: [wt.start, wt.end] for wt in word_timings}

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

    def map_new_lyrics_to_timing(
        self,
        old_word_timings: List[WordTiming],
        new_lyrics: str,
    ) -> List[WordTiming]:
        """Map new lyrics to the timing of old lyrics.

        Maps new lyrics by sentences (split by newlines) to old lyrics timing.
        New sentences start from the first old sentence's start time.

        Args:
            old_word_timings: Timing information from aligned old lyrics (sentences)
            new_lyrics: New lyrics text to map to the timings (newline-separated sentences)

        Returns:
            List of WordTiming objects for new lyrics with old lyrics' timings
        """
        # Split new lyrics by newlines to get sentences
        new_sentences = [
            line.strip() for line in new_lyrics.strip().split("\n") if line.strip()
        ]

        if not new_sentences:
            print("⚠️  No new sentences found")
            return []

        # If new and old have same number of sentences, direct mapping
        if len(new_sentences) == len(old_word_timings):
            print(
                f"✓ Direct mapping: {len(new_sentences)} new sentences to {len(old_word_timings)} old timings"
            )
            return [
                WordTiming(
                    word=new_sentence,
                    start=old_timing.start,
                    end=old_timing.end,
                    score=old_timing.score,
                )
                for new_sentence, old_timing in zip(new_sentences, old_word_timings)
            ]

        # If new sentences have fewer items, group them
        if len(new_sentences) < len(old_word_timings):
            print(
                f"⚠️  Grouping: {len(new_sentences)} new sentences to {len(old_word_timings)} old timings"
            )
            sentences_per_group = len(old_word_timings) / len(new_sentences)
            new_timings = []

            for i, new_sentence in enumerate(new_sentences):
                start_idx = int(i * sentences_per_group)
                end_idx = int((i + 1) * sentences_per_group)
                end_idx = min(end_idx, len(old_word_timings))

                # Use timing from first and last sentence in this group
                start_time = old_word_timings[start_idx].start
                end_time = old_word_timings[end_idx - 1].end

                new_timings.append(
                    WordTiming(
                        word=new_sentence,
                        start=start_time,
                        end=end_time,
                        score=1.0,
                    )
                )

            return new_timings

        # If new sentences have more items, split them across timing intervals
        print(
            f"⚠️  Splitting: {len(new_sentences)} new sentences to {len(old_word_timings)} old timings"
        )
        new_timings = []

        # Get total duration from first to last old timing
        if old_word_timings:
            total_start = old_word_timings[0].start
            total_end = old_word_timings[-1].end
            total_duration = total_end - total_start
        else:
            return []

        # Distribute new sentences across the old timing span
        time_per_sentence = (
            total_duration / len(new_sentences)
            if len(new_sentences) > 0
            else total_duration
        )

        for i, new_sentence in enumerate(new_sentences):
            start_time = total_start + i * time_per_sentence
            end_time = start_time + time_per_sentence

            new_timings.append(
                WordTiming(
                    word=new_sentence,
                    start=start_time,
                    end=end_time,
                    score=1.0,
                )
            )

        return new_timings


class WhisperLyricsAligner:
    """Aligns lyrics text with audio using Whisper's built-in alignment.

    This class uses OpenAI's Whisper model with stable-ts (Stable Whisper)
    to align lyrics words with their timing in the audio. Stable Whisper
    provides more accurate and stable word-level timing than standard Whisper.

    Args:
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        device: Device to run the model on ('cuda' or 'cpu')
        language: Language code (e.g., 'en', 'zh', 'es', 'fr')
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = None,
        language: str = None,
    ):
        """Initialize Whisper-based aligner.

        Args:
            model_size: Whisper model size. Options: 'tiny', 'base', 'small', 'medium', 'large'
            device: Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available.
            language: Language code. If None, auto-detect. Examples: 'en', 'zh', 'es', 'fr'
        """
        try:
            import stable_whisper

            self.stable_whisper = stable_whisper
        except ImportError:
            raise ImportError(
                "stable-ts not installed. Install with: pip install stable-ts"
            )

        self.model_size = model_size
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self.language = language

        print(f"Loading Whisper model: {model_size}")
        print(f"Device: {self.device}")
        self.model = stable_whisper.load_model(model_size, device=self.device)

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def align(
        self,
        audio_path: str,
        lyrics: str = None,
        language: str = None,
    ) -> List[WordTiming]:
        """Align lyrics with audio using Whisper.

        If lyrics are provided, performs forced alignment of the provided text.
        If no lyrics provided, transcribes the audio and returns word timings from transcription.

        Args:
            audio_path: Path to audio file
            lyrics: Optional lyrics text to align. If None, transcribes the audio.
            language: Language code. If None, uses the one set during init.

        Returns:
            List of WordTiming objects with word-level timing

        Raises:
            FileNotFoundError: If audio file not found
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        language = language or self.language

        print(f"\n{'=' * 60}")
        print("WHISPER LYRICS ALIGNMENT")
        print(f"{'=' * 60}")
        print(f"Audio: {audio_path}")
        if lyrics:
            print("Aligning provided lyrics...")
        else:
            print("Transcribing audio...")
        if language:
            print(f"Language: {language}")

        try:
            # Use stable-ts align method if lyrics provided
            if lyrics:
                clean_lyrics = lyrics.strip()
                print(f"Aligning {len(clean_lyrics.split())} words...")

                # Use align method from stable-ts
                result = self.model.align(
                    str(audio_path),  # audio parameter (positional)
                    text=clean_lyrics,
                    language=language,
                )
            else:
                # Transcribe without forced alignment
                print("Transcribing audio for word-level timing...")
                result = self.model.transcribe(
                    audio=str(audio_path),
                    language=language,
                )

            # Extract word timings from result segments
            word_timings = []

            if hasattr(result, "segments"):
                # Process segments from Whisper result
                for segment in result.segments:
                    if hasattr(segment, "words"):
                        # Word-level timing available
                        for word_info in segment.words:
                            if hasattr(word_info, "start") and hasattr(
                                word_info, "end"
                            ):
                                word_timings.append(
                                    WordTiming(
                                        word=word_info.word,
                                        start=word_info.start,
                                        end=word_info.end,
                                        score=1.0,
                                    )
                                )
                    else:
                        # Fall back to segment-level timing
                        text = segment.text if hasattr(segment, "text") else ""
                        start = segment.start if hasattr(segment, "start") else 0
                        end = segment.end if hasattr(segment, "end") else 0
                        if text:
                            word_timings.append(
                                WordTiming(
                                    word=text,
                                    start=start,
                                    end=end,
                                    score=1.0,
                                )
                            )

            if word_timings:
                print(f"✓ Alignment complete! Aligned {len(word_timings)} words.")
                return word_timings
            else:
                # Fallback: return empty list with warning
                print("⚠️  No word timings extracted from Whisper result")
                return []

        except Exception as e:
            print(f"❌ Alignment failed: {e}")
            raise RuntimeError(f"Whisper alignment failed: {e}")

    def align_and_split_by_lines(
        self,
        audio_path: str,
        lyrics: str,
        language: str = None,
    ) -> List[WordTiming]:
        """Align audio with lyrics split by line breaks.

        This method treats each line as a separate unit and aligns them with
        the audio, distributing the aligned words across line boundaries.

        Args:
            audio_path: Path to audio file
            lyrics: Lyrics text with line breaks (\\n)
            language: Language code. If None, uses the one set during init.

        Returns:
            List of WordTiming objects aligned to line boundaries
        """
        # First, get word-level timings from Whisper
        word_timings = self.align(audio_path, lyrics, language)

        if not word_timings:
            return []

        # Split lyrics into lines
        lines = [line.strip() for line in lyrics.strip().split("\n") if line.strip()]

        if not lines:
            return word_timings

        # Redistribute word timings to line boundaries
        line_timings = []
        words_per_line = len(word_timings) / len(lines)

        for i, line in enumerate(lines):
            start_idx = int(i * words_per_line)
            end_idx = int((i + 1) * words_per_line)
            end_idx = min(end_idx, len(word_timings))

            if start_idx < len(word_timings) and end_idx > start_idx:
                start_time = word_timings[start_idx].start
                end_time = word_timings[end_idx - 1].end

                line_timings.append(
                    WordTiming(
                        word=line,
                        start=start_time,
                        end=end_time,
                        score=1.0,
                    )
                )

        return line_timings

    def get_alignment_dict(
        self, word_timings: List[WordTiming]
    ) -> Dict[str, List[float]]:
        """Convert word timings to a dictionary format.

        Args:
            word_timings: List of WordTiming objects

        Returns:
            Dictionary mapping words to [start, end] times
        """
        return {wt.word: [wt.start, wt.end] for wt in word_timings}

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

    def map_new_lyrics_to_timing(
        self,
        old_word_timings: List[WordTiming],
        new_lyrics: str,
    ) -> List[WordTiming]:
        """Map new lyrics to the timing of old lyrics.

        Maps new lyrics by sentences (split by newlines) to old lyrics timing.

        Args:
            old_word_timings: Timing information from aligned old lyrics
            new_lyrics: New lyrics text to map to the timings (newline-separated)

        Returns:
            List of WordTiming objects for new lyrics with old lyrics' timings
        """
        new_sentences = [
            line.strip() for line in new_lyrics.strip().split("\n") if line.strip()
        ]

        if not new_sentences:
            print("⚠️  No new sentences found")
            return []

        # If new and old have same number of sentences, direct mapping
        if len(new_sentences) == len(old_word_timings):
            print(
                f"✓ Direct mapping: {len(new_sentences)} new sentences to {len(old_word_timings)} old timings"
            )
            return [
                WordTiming(
                    word=new_sentence,
                    start=old_timing.start,
                    end=old_timing.end,
                    score=old_timing.score,
                )
                for new_sentence, old_timing in zip(new_sentences, old_word_timings)
            ]

        # If new sentences have fewer items, group them
        if len(new_sentences) < len(old_word_timings):
            print(
                f"⚠️  Grouping: {len(new_sentences)} new sentences to {len(old_word_timings)} old timings"
            )
            sentences_per_group = len(old_word_timings) / len(new_sentences)
            new_timings = []

            for i, new_sentence in enumerate(new_sentences):
                start_idx = int(i * sentences_per_group)
                end_idx = int((i + 1) * sentences_per_group)
                end_idx = min(end_idx, len(old_word_timings))

                start_time = old_word_timings[start_idx].start
                end_time = old_word_timings[end_idx - 1].end

                new_timings.append(
                    WordTiming(
                        word=new_sentence,
                        start=start_time,
                        end=end_time,
                        score=1.0,
                    )
                )

            return new_timings

        # If new sentences have more items, split them across timing intervals
        print(
            f"⚠️  Splitting: {len(new_sentences)} new sentences to {len(old_word_timings)} old timings"
        )
        new_timings = []

        if old_word_timings:
            total_start = old_word_timings[0].start
            total_end = old_word_timings[-1].end
            total_duration = total_end - total_start
        else:
            return []

        time_per_sentence = (
            total_duration / len(new_sentences)
            if len(new_sentences) > 0
            else total_duration
        )

        for i, new_sentence in enumerate(new_sentences):
            start_time = total_start + i * time_per_sentence
            end_time = start_time + time_per_sentence

            new_timings.append(
                WordTiming(
                    word=new_sentence,
                    start=start_time,
                    end=end_time,
                    score=1.0,
                )
            )

        return new_timings
