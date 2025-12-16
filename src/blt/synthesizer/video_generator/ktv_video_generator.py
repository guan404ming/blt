"""KTV (Karaoke) video generation with dual-track subtitles.

This module provides functionality to generate KTV-style videos with
synchronized subtitles showing both main lyrics and phonetic translations.
"""

import datetime
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any


class KTVVideoGenerator:
    """Generates KTV (Karaoke) videos with dual-track subtitles.

    Creates videos with synchronized subtitles showing both the main lyrics
    and optional sub-lyrics (e.g., phonetic translation, soramimi).

    Args:
        font_name: Font name for subtitles (default: "Noto Sans CJK TC")
        main_font_size: Font size for main lyrics (default: 60)
        sub_font_size: Font size for sub lyrics (default: 40)
    """

    def __init__(
        self,
        font_name: str = "Noto Sans CJK TC",
        main_font_size: int = 60,
        sub_font_size: int = 40,
    ):
        """Initialize KTVVideoGenerator."""
        self.font_name = font_name
        self.main_font_size = main_font_size
        self.sub_font_size = sub_font_size

        print("Initializing KTVVideoGenerator")
        print(f"  Font: {font_name}")
        print(f"  Main font size: {main_font_size}")
        print(f"  Sub font size: {sub_font_size}")

    def generate(
        self,
        video_path: str,
        audio_path: str,
        main_lyrics: str,
        alignment_result: Any,
        output_path: str,
        sub_lyrics: Optional[str] = None,
    ) -> str:
        """Generate KTV video with subtitles.

        Args:
            video_path: Path to input video (e.g., lip-synced video)
            audio_path: Path to audio file
            main_lyrics: Main lyrics text (newline-separated)
            alignment_result: Alignment result from WhisperLyricsAligner (List[WordTiming])
                            or stable-whisper result object
            output_path: Path to save the output KTV video
            sub_lyrics: Optional sub-lyrics text (e.g., phonetic translation)

        Returns:
            Path to the generated KTV video

        Raises:
            FileNotFoundError: If input files don't exist
            RuntimeError: If video generation fails
        """
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        # Validate input files
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("KTV VIDEO GENERATION")
        print("=" * 60)
        print(f"Video: {video_path}")
        print(f"Audio: {audio_path}")
        print(f"Output: {output_path}")
        print()

        # Generate subtitle file
        subtitle_path = output_path.with_suffix(".ass")

        if sub_lyrics:
            # Dual-track subtitles
            print("🎯 Generating dual-track subtitles...")
            segments = self._realign_dual_tracks(
                alignment_result, main_lyrics, sub_lyrics
            )
            self._generate_dual_track_ass(segments, subtitle_path)
        else:
            # Single-track subtitles
            print("🎯 Generating single-track subtitles...")
            segments = self._extract_segments(alignment_result, main_lyrics)
            self._generate_single_track_ass(segments, subtitle_path)

        # Combine video with subtitles
        print("🎬 Rendering final video...")
        return self._render_video(video_path, audio_path, subtitle_path, output_path)

    def _format_time(self, seconds: float) -> str:
        """Format seconds to ASS timestamp format (H:MM:SS.CC)."""
        dt = datetime.datetime.fromtimestamp(seconds, tz=datetime.UTC)
        return dt.strftime("%-H:%M:%S.%f")[:-4]

    def _realign_dual_tracks(
        self,
        alignment_result: Any,
        main_lyrics: str,
        sub_lyrics: str,
    ) -> List[Dict[str, Any]]:
        """Realign dual-track lyrics based on word-level timing.

        Args:
            alignment_result: Either stable-whisper result or List[WordTiming]
            main_lyrics: Main lyrics text (newline-separated)
            sub_lyrics: Sub lyrics text (newline-separated)

        Returns:
            List of aligned segments with dual-track lyrics
        """
        print("   🧩 Calculating dual-track alignment...")

        # Extract all words from alignment result
        # Handle both stable-whisper result object and List[WordTiming]
        all_words = []
        if isinstance(alignment_result, list):
            # List[WordTiming] format from WhisperLyricsAligner
            all_words = alignment_result
        elif hasattr(alignment_result, "segments"):
            # stable-whisper result object
            for seg in alignment_result.segments:
                all_words.extend(seg.words)
        else:
            raise ValueError(
                "alignment_result must be either a stable-whisper result object or List[WordTiming]"
            )

        # Split lyrics into lines
        main_lines = [
            line.strip() for line in main_lyrics.strip().split("\n") if line.strip()
        ]
        sub_lines = [
            line.strip() for line in sub_lyrics.strip().split("\n") if line.strip()
        ]

        min_len = min(len(main_lines), len(sub_lines))
        new_segments = []
        word_index = 0

        for i in range(min_len):
            main_line = main_lines[i]
            sub_line = sub_lines[i]
            current_segment_words = []

            # Calculate how many characters we need to grab
            clean_line_len = len(main_line.replace(" ", ""))
            grabbed_chars = 0

            # Grab words until we have enough characters for this line
            while grabbed_chars < clean_line_len and word_index < len(all_words):
                word_obj = all_words[word_index]
                current_segment_words.append(word_obj)
                grabbed_chars += len(word_obj.word.strip())
                word_index += 1

            if current_segment_words:
                new_segments.append(
                    {
                        "start": current_segment_words[0].start,
                        "end": current_segment_words[-1].end,
                        "words": current_segment_words,
                        "text_main": main_line,
                        "text_sub": sub_line,
                    }
                )

        return new_segments

    def _extract_segments(
        self,
        alignment_result: Any,
        main_lyrics: str,
    ) -> List[Dict[str, Any]]:
        """Extract segments from alignment result for single-track subtitles.

        Args:
            alignment_result: Either stable-whisper result or List[WordTiming]
            main_lyrics: Main lyrics text (newline-separated)

        Returns:
            List of aligned segments
        """
        # Extract all words
        # Handle both stable-whisper result object and List[WordTiming]
        all_words = []
        if isinstance(alignment_result, list):
            # List[WordTiming] format from WhisperLyricsAligner
            all_words = alignment_result
        elif hasattr(alignment_result, "segments"):
            # stable-whisper result object
            for seg in alignment_result.segments:
                all_words.extend(seg.words)
        else:
            raise ValueError(
                "alignment_result must be either a stable-whisper result object or List[WordTiming]"
            )

        # Split lyrics into lines
        lines = [
            line.strip() for line in main_lyrics.strip().split("\n") if line.strip()
        ]

        segments = []
        word_index = 0

        for line in lines:
            current_words = []
            clean_line_len = len(line.replace(" ", ""))
            grabbed_chars = 0

            while grabbed_chars < clean_line_len and word_index < len(all_words):
                word_obj = all_words[word_index]
                current_words.append(word_obj)
                grabbed_chars += len(word_obj.word.strip())
                word_index += 1

            if current_words:
                segments.append(
                    {
                        "start": current_words[0].start,
                        "end": current_words[-1].end,
                        "words": current_words,
                        "text_main": line,
                    }
                )

        return segments

    def _generate_dual_track_ass(
        self,
        segments: List[Dict[str, Any]],
        ass_path: Path,
    ) -> None:
        """Generate dual-track ASS subtitle file.

        Args:
            segments: List of aligned segments with dual-track text
            ass_path: Path to save ASS file
        """
        try:
            from opencc import OpenCC

            cc = OpenCC("s2t")
            has_opencc = True
        except ImportError:
            print("   ⚠️ OpenCC not available, skipping traditional Chinese conversion")
            cc = None
            has_opencc = False

        # ASS header
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: MainStyle,{self.font_name},{self.main_font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,65,1
Style: SubStyle,{self.font_name},{self.sub_font_size},&H0000FFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,15,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        events = []

        for seg in segments:
            start_time = self._format_time(seg["start"])
            end_time = self._format_time(seg["end"])

            # Main lyrics with karaoke timing
            k_text_main = ""
            for i, word in enumerate(seg["words"]):
                duration = int((word.end - word.start) * 100)
                raw_word = word.word

                # Check if word has Chinese characters
                has_chinese = any("\u4e00" <= char <= "\u9fff" for char in raw_word)

                if has_chinese and has_opencc:
                    word_text = cc.convert(raw_word.strip())
                else:
                    clean_word = raw_word.strip()
                    if i == 0:
                        word_text = clean_word
                    else:
                        word_text = " " + clean_word

                if word_text:
                    k_text_main += f"{{\\k{duration}}}{word_text}"

            if k_text_main:
                events.append(
                    f"Dialogue: 0,{start_time},{end_time},MainStyle,,0,0,0,,{{\\df2}}{k_text_main}"
                )

            # Sub lyrics (simple display, no karaoke timing)
            sub_text = seg["text_sub"]
            if sub_text:
                total_duration = int((seg["end"] - seg["start"]) * 100)
                events.append(
                    f"Dialogue: 0,{start_time},{end_time},SubStyle,,0,0,0,,{{\\df2}}{{\\k{total_duration}}}{sub_text}"
                )

        # Write ASS file
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(events))

        print(f"   ✨ Dual-track ASS subtitle file generated: {ass_path}")

    def _generate_single_track_ass(
        self,
        segments: List[Dict[str, Any]],
        ass_path: Path,
    ) -> None:
        """Generate single-track ASS subtitle file.

        Args:
            segments: List of aligned segments
            ass_path: Path to save ASS file
        """
        try:
            from opencc import OpenCC

            cc = OpenCC("s2t")
            has_opencc = True
        except ImportError:
            print("   ⚠️ OpenCC not available, skipping traditional Chinese conversion")
            cc = None
            has_opencc = False

        # ASS header
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: MainStyle,{self.font_name},{self.main_font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,65,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        events = []

        for seg in segments:
            start_time = self._format_time(seg["start"])
            end_time = self._format_time(seg["end"])

            # Main lyrics with karaoke timing
            k_text = ""
            for i, word in enumerate(seg["words"]):
                duration = int((word.end - word.start) * 100)
                raw_word = word.word

                # Check if word has Chinese characters
                has_chinese = any("\u4e00" <= char <= "\u9fff" for char in raw_word)

                if has_chinese and has_opencc:
                    word_text = cc.convert(raw_word.strip())
                else:
                    clean_word = raw_word.strip()
                    if i == 0:
                        word_text = clean_word
                    else:
                        word_text = " " + clean_word

                if word_text:
                    k_text += f"{{\\k{duration}}}{word_text}"

            if k_text:
                events.append(
                    f"Dialogue: 0,{start_time},{end_time},MainStyle,,0,0,0,,{{\\df2}}{k_text}"
                )

        # Write ASS file
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(events))

        print(f"   ✨ ASS subtitle file generated: {ass_path}")

    def _render_video(
        self,
        video_path: Path,
        audio_path: Path,
        subtitle_path: Path,
        output_path: Path,
    ) -> str:
        """Render final video with subtitles and audio.

        Args:
            video_path: Path to input video
            audio_path: Path to audio file
            subtitle_path: Path to ASS subtitle file
            output_path: Path to save output video

        Returns:
            Path to the generated video

        Raises:
            RuntimeError: If FFmpeg rendering fails
        """
        # Convert to absolute paths
        video_path = video_path.resolve()
        audio_path = audio_path.resolve()
        subtitle_path = subtitle_path.resolve()
        output_path = output_path.resolve()

        # FFmpeg command to combine video, audio, and subtitles
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-v",
            "error",  # Only show errors
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-vf",
            f"ass={subtitle_path.name}",  # Use filename only for ASS filter
            "-map",
            "0:v",  # Video from first input
            "-map",
            "1:a",  # Audio from second input
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",  # Stop at shortest stream
            str(output_path),
        ]

        try:
            # Run FFmpeg from the subtitle directory
            import os

            original_cwd = Path.cwd()
            os.chdir(subtitle_path.parent)

            subprocess.run(cmd, check=True, capture_output=True, text=True)

            os.chdir(original_cwd)

            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print()
                print("=" * 60)
                print("✓ SUCCESS!")
                print("=" * 60)
                print(f"KTV video created: {output_path}")
                print(f"Size: {size_mb:.2f} MB")
                print()
                return str(output_path)
            else:
                raise RuntimeError("Output video was not created")

        except subprocess.CalledProcessError as e:
            os.chdir(original_cwd)
            print()
            print("=" * 60)
            print("❌ FAILED")
            print("=" * 60)
            print("FFmpeg error:")
            print(e.stderr)
            raise RuntimeError(f"FFmpeg rendering failed: {e.stderr}")

        except Exception as e:
            os.chdir(original_cwd)
            raise RuntimeError(f"Video rendering failed: {e}")
