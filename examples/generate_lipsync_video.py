"""Example: Generate lip-synced video using LipSyncedVideoGenerator.

This example demonstrates how to use LipSyncedVideoGenerator to create
a lip-synced video from a face image and audio file.

Requirements:
1. Wav2Lip repository at: src/blt/synthesizer/Wav2Lip
2. Checkpoint at: Wav2Lip/checkpoints/wav2lip_gan.pth
3. Face detection model at: Wav2Lip/face_detection/detection/sfd/s3fd.pth

Usage:
    uv run examples/generate_lipsync_video.py
"""

from blt.synthesizer import LipSyncedVideoGenerator


def main():
    # Input files
    face_image = "assets/godtone.jpg"
    audio_file = "assets/擱淺_mixed.wav"

    # Output video file
    output_video = "assets/lipsync-擱淺.mp4"

    print("=" * 60)
    print("LIP-SYNC VIDEO GENERATION EXAMPLE")
    print("=" * 60)
    print(f"Face image: {face_image}")
    print(f"Audio file: {audio_file}")
    print(f"Output: {output_video}")
    print()

    try:
        # Create generator
        generator = LipSyncedVideoGenerator()

        # Generate lip-synced video
        result_path = generator.generate(
            face_path=face_image,
            audio_path=audio_file,
            output_path=output_video,
            resize_factor=1,  # 1 = best quality, 2 = faster
        )

        print()
        print("=" * 60)
        print("GENERATION RESULTS")
        print("=" * 60)
        print(f"Lip-synced video: {result_path}")
        print()
        print("✓ Generation complete!")
        print()
        print("You can now use this video for:")
        print("  - Final output")
        print("  - KTV video generation with subtitles")
        print("  - Further video processing")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. Input files exist in the assets directory")
        print("2. Wav2Lip is installed in src/blt/synthesizer/Wav2Lip")
        print("3. Required model files are downloaded")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
