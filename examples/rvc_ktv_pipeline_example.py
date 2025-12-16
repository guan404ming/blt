"""Example: Complete RVC KTV pipeline for voice conversion and video generation.

This example demonstrates the end-to-end RVCKTVPipeline that:
1. Separates vocals from instrumental
2. Converts vocals using RVC model
3. Mixes converted vocals with instrumental
4. Generates lip-synced video
5. Creates KTV video with dual-track subtitles

Requirements:
1. RVC model files (model.pth, model.index)
2. Wav2Lip for lip-sync generation
3. FFmpeg for video processing

Usage:
    uv run examples/rvc_ktv_pipeline_example.py
"""

from blt.pipeline import RVCKTVPipeline


def main():
    # Input files
    audio_file = "assets/擱淺.mp3"
    rvc_model = "assets/model.pth"
    rvc_index = "assets/model.index"
    face_image = "assets/godtone.jpg"

    # Read lyrics from file
    lyrics_file = "assets/lyrics-擱淺.txt"
    try:
        with open(lyrics_file, "r", encoding="utf-8") as f:
            main_lyrics = f.read()
    except FileNotFoundError:
        print(f"❌ Lyrics file not found: {lyrics_file}")
        return

    # Sub lyrics (phonetic translation)
    sub_lyrics = """wo zhi neng yong yuan du zhe dui bai
du zhe wo gei ni de shang hai
wo yuan liang bu liao wo
jiu qing ni dang zuo wo yi bu zai
wo zheng kai shuang yan kan zhe kong bai
wang ji ni dui wo de qi dai
du wan le yi lai
wo hen kuai jiu li kai
"""

    # Output name
    output_name = "擱淺"

    print("=" * 60)
    print("RVC KTV PIPELINE EXAMPLE")
    print("=" * 60)
    print(f"Audio: {audio_file}")
    print(f"RVC Model: {rvc_model}")
    print(f"Face: {face_image}")
    print(f"Output: rvc_ktv_output/{output_name}/")
    print()

    try:
        # Initialize pipeline
        pipeline = RVCKTVPipeline(
            separator_model="htdemucs",
            whisper_model="medium",
            output_dir="rvc_ktv_output",
        )

        # Run complete pipeline
        results = pipeline.run(
            audio_path=audio_file,
            rvc_model_path=rvc_model,
            rvc_index_path=rvc_index,
            face_path=face_image,
            main_lyrics=main_lyrics,
            sub_lyrics=sub_lyrics,
            lyrics_language="zh",
            output_name=output_name,
            pitch_shift=0,  # Pitch shift (male to female +12, female to male -12)
            vocal_volume=1.0,
            instrumental_volume=0.8,
        )

        print()
        print("=" * 60)
        print("PIPELINE RESULTS")
        print("=" * 60)
        print(f"✓ Mixed audio: {results['mixed_audio']}")
        print(f"✓ Lip-sync video: {results['lipsync_video']}")
        print(f"✓ KTV video: {results['ktv_video']}")
        print()
        print("All outputs saved to: rvc_ktv_output/擱淺/")
        print()
        print("Pipeline completed successfully! 🎉")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. Input audio file exists")
        print("2. RVC model files exist (model.pth, model.index)")
        print("3. Face image/video exists")
        print("4. Lyrics file exists")
        print("5. Wav2Lip is installed in src/blt/synthesizer/Wav2Lip")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
