"""Wav2Lip Example - Simple Lip-Sync Generation

This example demonstrates Wav2Lip usage to create a lip-synced video from:
- A face image (godtone.jpg)
- An audio file (擱淺.mp3 or any audio)

Requirements:
1. Wav2Lip repository cloned at: /home/gmchiu/Documents/GitHub/blt/Wav2Lip
2. Checkpoint at: Wav2Lip/checkpoints/wav2lip_gan.pth
3. Face detection model at: Wav2Lip/face_detection/detection/sfd/s3fd.pth

Usage:
    python examples/wav2lip_example.py
"""

import sys
import subprocess
from pathlib import Path


def main():
    # Paths
    examples_dir = Path(__file__).parent
    wav2lip_dir = Path("/home/gmchiu/Documents/GitHub/blt/Wav2Lip")

    face_image = examples_dir / "godtone.jpg"
    audio_file = examples_dir / "擱淺.mp3"
    output_video = examples_dir / "wav2lip_output.mp4"

    checkpoint = "checkpoints/wav2lip_gan.pth"

    print("=" * 60)
    print("WAV2LIP LIP-SYNC EXAMPLE")
    print("=" * 60)
    print(f"Face image: {face_image}")
    print(f"Audio file: {audio_file}")
    print(f"Output: {output_video}")
    print()

    # Check files exist
    if not face_image.exists():
        print(f"❌ Face image not found: {face_image}")
        return

    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        return

    if not wav2lip_dir.exists():
        print(f"❌ Wav2Lip not found at: {wav2lip_dir}")
        print("   Please clone: git clone https://github.com/Rudrabha/Wav2Lip.git")
        return

    print("✅ Starting Wav2Lip...")
    print()

    # Change to Wav2Lip directory and run
    print("🎬 Running Wav2Lip...")
    print(f"   Command: python inference.py --checkpoint_path {checkpoint}")
    print(f"            --face {face_image.resolve()}")
    print(f"            --audio {audio_file.resolve()}")
    print(f"            --outfile {output_video.resolve()}")
    print()

    import os

    original_cwd = Path.cwd()

    try:
        os.chdir(wav2lip_dir)

        cmd = [
            sys.executable,
            "inference.py",
            "--checkpoint_path",
            checkpoint,
            "--face",
            str(face_image.resolve()),
            "--audio",
            str(audio_file.resolve()),
            "--outfile",
            str(output_video.resolve()),
            "--resize_factor",
            "1",  # 1 = best quality, 2 = faster
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        os.chdir(original_cwd)

        if result.returncode == 0 and output_video.exists():
            size_mb = output_video.stat().st_size / (1024 * 1024)
            print()
            print("=" * 60)
            print("✅ SUCCESS!")
            print("=" * 60)
            print(f"Video created: {output_video}")
            print(f"Size: {size_mb:.2f} MB")
            print()
            print("You can now play the video to see the lip-sync!")
        else:
            print()
            print("=" * 60)
            print("❌ FAILED")
            print("=" * 60)
            print("Error output:")
            print(result.stderr)
            if result.stdout:
                print("\nStandard output:")
                print(result.stdout)

    except subprocess.TimeoutExpired:
        os.chdir(original_cwd)
        print("❌ Timeout: Wav2Lip took too long (>10 minutes)")

    except Exception as e:
        os.chdir(original_cwd)
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
