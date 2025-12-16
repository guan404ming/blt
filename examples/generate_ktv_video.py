"""Example: Generate KTV video with dual-track subtitles using KTVVideoGenerator.

This example demonstrates how to use KTVVideoGenerator to create a KTV-style
video with synchronized subtitles showing both main lyrics and phonetic translation.

Requirements:
1. stable-whisper for lyrics alignment
2. FFmpeg for video rendering
3. (Optional) OpenCC for traditional Chinese conversion

Usage:
    uv run examples/generate_ktv_video.py
"""

from blt.synthesizer import KTVVideoGenerator, WhisperLyricsAligner


def main():
    # Input files
    lipsync_video = "assets/lipsync-擱淺.mp4"
    audio_file = "assets/擱淺_mixed.wav"

    # Output KTV video
    output_video = "assets/ktv-擱淺.mp4"

    # Read main lyrics from file
    lyrics_file = "assets/lyrics-擱淺.txt"

    try:
        with open(lyrics_file, "r", encoding="utf-8") as f:
            main_lyrics = f.read()
    except FileNotFoundError:
        print(f"❌ Lyrics file not found: {lyrics_file}")
        return

    # Sub lyrics (phonetic translation / soramimi)
    # TODO: Create a pinyin/soramimi version of the lyrics
    sub_lyrics = """wo zhi neng yong yuan du zhe dui bai
du zhe wo gei ni de shang hai
wo yuan liang bu liao wo
jiu qing ni dang zuo wo yi bu zai
wo zheng kai shuang yan kan zhe kong bai
wang ji ni dui wo de qi dai
du wan le yi lai
wo hen kuai jiu li kai
"""

    print("=" * 60)
    print("KTV VIDEO GENERATION EXAMPLE")
    print("=" * 60)
    print(f"Input video: {lipsync_video}")
    print(f"Input audio: {audio_file}")
    print(f"Output: {output_video}")
    print()

    try:
        # Step 1: Align lyrics with audio
        print("Step 1: Aligning lyrics with audio...")
        aligner = WhisperLyricsAligner(model_size="medium")
        alignment_result = aligner.align(
            audio_path=audio_file,
            lyrics=main_lyrics,
            language="zh",  # Chinese
        )
        print("✓ Lyrics alignment complete!")
        print()

        # Step 2: Generate KTV video with dual-track subtitles
        print("Step 2: Generating KTV video...")
        generator = KTVVideoGenerator(
            font_name="Noto Sans CJK TC",
            main_font_size=60,
            sub_font_size=40,
        )

        result_path = generator.generate(
            video_path=lipsync_video,
            audio_path=audio_file,
            main_lyrics=main_lyrics,
            alignment_result=alignment_result,
            output_path=output_video,
            sub_lyrics=sub_lyrics,  # Optional: set to None for single-track
        )

        print()
        print("=" * 60)
        print("GENERATION RESULTS")
        print("=" * 60)
        print(f"KTV video: {result_path}")
        print()
        print("✓ Generation complete!")
        print()
        print("Features:")
        print("  - Synchronized karaoke-style subtitles")
        print("  - Dual-track lyrics (main + phonetic)")
        print("  - Professional KTV video output")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. Input files exist in the assets directory")
        print("2. Run wav2lip_example.py first to generate lipsync-擱淺.mp4")
        print("3. FFmpeg is installed on your system")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
