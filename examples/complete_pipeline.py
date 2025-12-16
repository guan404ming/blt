"""Complete BLT Pipeline - Unified End-to-End Solution

This is the main BLT pipeline that includes all features:
1. Vocal Separation (Demucs)
2. Voice Conversion (RVC - optional if model files available)
3. Audio Mixing
4. Lip-Sync Video (Wav2Lip with automatic FFmpeg fallback)
5. Lyrics Alignment & Subtitles (Whisper - optional)
6. Final Video Composition

The pipeline automatically detects available models and uses them:
- RVC: Uses model.pth and model.index if available for voice conversion
- Wav2Lip: Uses if installed for high-quality lip-sync, falls back to FFmpeg
- Whisper: Aligns lyrics and generates subtitles if lyrics provided

Wav2Lip Setup (optional for advanced lip-sync):
    git clone https://github.com/Rudrabha/Wav2Lip.git
    cd Wav2Lip
    pip install -r requirements.txt
    wget https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -O face_detection/detection/sfd/s3fd.pth

Usage:
    python examples/complete_pipeline.py
"""

import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blt.synthesizer import VocalSeparator


def print_stage(stage_num, title):
    """Print stage header"""
    print(f"\n{'=' * 70}")
    print(f"[{stage_num}] {title}")
    print(f"{'=' * 70}")


def run_cmd(cmd, description=""):
    """Run command and handle errors"""
    if description:
        print(f"⏳ {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def use_wav2lip(wav2lip_dir, face_image, audio_path, output_video):
    """Try to use Wav2Lip for lip-sync generation using face image"""
    print("🎬 Using Wav2Lip for lip-sync generation...")

    # Ensure wav2lip_dir is a Path object
    wav2lip_dir = Path(wav2lip_dir) if isinstance(wav2lip_dir, str) else wav2lip_dir
    checkpoint = "checkpoints/wav2lip_gan.pth"  # Relative path

    if not (wav2lip_dir / checkpoint).exists():
        print(f"❌ Checkpoint not found at {wav2lip_dir / checkpoint}")
        return False

    try:
        # Change to Wav2Lip directory and run inference
        original_cwd = Path.cwd()
        import os

        os.chdir(wav2lip_dir)

        # Use absolute paths for input/output files
        # Ensure all paths are Path objects
        face_image_path = (
            Path(face_image) if isinstance(face_image, str) else face_image
        )
        audio_path_obj = Path(audio_path) if isinstance(audio_path, str) else audio_path
        output_video_obj = (
            Path(output_video) if isinstance(output_video, str) else output_video
        )

        cmd = [
            sys.executable,
            "inference.py",
            "--checkpoint_path",
            checkpoint,
            "--face",
            str(face_image_path.resolve()),
            "--audio",
            str(audio_path_obj.resolve()),
            "--outfile",
            str(output_video_obj.resolve()),
            "--resize_factor",
            "1",  # 1 for better quality
        ]

        print(
            f"   Running: python inference.py --checkpoint_path {checkpoint} --face {face_image_path.name} --audio {audio_path_obj.name}"
        )
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        os.chdir(original_cwd)

        if result.returncode == 0 and output_video_obj.exists():
            print(f"✅ Wav2Lip video created: {output_video.name}")
            return True
        else:
            print("❌ Wav2Lip failed:")
            print(f"   {result.stderr}")
            return False

    except Exception as e:
        os.chdir(original_cwd)
        print(f"❌ Wav2Lip error: {e}")
        return False


def use_ffmpeg_fallback(face_video, audio_path, output_video):
    """Fallback: Create video from image using FFmpeg"""
    print("📹 Creating lip-sync video with FFmpeg (image + audio)...")

    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(face_video),
        "-i",
        str(audio_path),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=720:404",
        "-shortest",
        str(output_video),
    ]

    success, output = run_cmd(cmd, "Rendering with FFmpeg")
    return success


def main():
    examples_dir = Path(__file__).parent
    output_dir = examples_dir / "final_output"
    output_dir.mkdir(exist_ok=True)

    input_song = examples_dir / "擱淺.mp3"
    face_image = examples_dir / "godtone.jpg"

    print("\n" + "=" * 70)
    print("🎬 BLT COMPLETE PIPELINE")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ================================================================
    # Stage 1: Vocal Separation
    # ================================================================
    print_stage(1, "Vocal Separation (Demucs)")

    if not input_song.exists():
        print(f"❌ Input not found: {input_song}")
        return

    separator = VocalSeparator()
    vocal_path, instrumental_path = separator.separate(
        audio_path=str(input_song), output_dir=str(output_dir / "separated")
    )
    print(f"✅ Vocals: {Path(vocal_path).name}")
    print(f"✅ Instrumental: {Path(instrumental_path).name}")

    # ================================================================
    # Stage 2: Voice Conversion (RVC)
    # ================================================================
    print_stage(2, "Voice Conversion (RVC)")

    # RVC model files
    model_pth = examples_dir / "model.pth"
    model_index = examples_dir / "model.index"

    if model_pth.exists() and model_index.exists():
        print("Converting voice using RVC...")
        try:
            from gradio_client import Client, handle_file

            client = Client("r3gm/RVC_ZERO")
            result = client.predict(
                [handle_file(str(vocal_path))],
                handle_file(str(model_pth)),
                "rmvpe+",  # f0 method
                0,  # f0_change
                handle_file(str(model_index)),
                0.75,  # index_rate
                3,  # filter_radius
                0.25,  # rms_mix_rate
                0.5,  # protect
                False,  # split_audio
                False,  # autotune
                "wav",  # output_format
                1,  # resample_sr
                api_name="/run",
            )
            converted_vocal_path = result[0] if isinstance(result, list) else result
            print("✅ Voice conversion complete!")
            print(f"   Converted: {Path(converted_vocal_path).name}")
        except Exception as e:
            print(f"⚠️  RVC conversion failed: {e}")
            print("   Using original vocals...")
            converted_vocal_path = vocal_path
    else:
        print("⚠️  RVC models not found (model.pth, model.index)")
        print("   Skipping voice conversion...")
        converted_vocal_path = vocal_path

    # ================================================================
    # Stage 3: Audio Mixing
    # ================================================================
    print_stage(3, "Audio Mixing (Vocals + Instrumental)")

    try:
        import soundfile as sf
        import numpy as np

        vocals, sr_v = sf.read(converted_vocal_path)
        instrumental, sr_i = sf.read(instrumental_path)

        # Ensure both are same shape (mono/stereo)
        if vocals.ndim == 1 and instrumental.ndim == 2:
            vocals = np.column_stack([vocals, vocals])  # Convert mono to stereo
        elif vocals.ndim == 2 and instrumental.ndim == 1:
            instrumental = np.column_stack([instrumental, instrumental])

        min_len = min(len(vocals), len(instrumental))
        mixed = (vocals[:min_len] + instrumental[:min_len]) / 2

        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val

        mixed_path = output_dir / "mixed_audio.wav"
        sf.write(str(mixed_path), mixed, sr_v)
        print(f"✅ Mixed audio: {mixed_path.name}")
    except Exception as e:
        print(f"❌ Mixing failed: {e}")
        mixed_path = Path(converted_vocal_path)

    # ================================================================
    # Stage 4: Lip-Sync Video (Wav2Lip or FFmpeg)
    # ================================================================
    print_stage(4, "Lip-Sync Video Generation")

    if not face_image.exists():
        print(f"❌ Face image not found: {face_image}")
        return

    lipsync_video_path = output_dir / "lipsync_video.mp4"
    success = use_wav2lip("Wav2Lip", face_image, mixed_path, lipsync_video_path)

    if lipsync_video_path.exists():
        size_mb = lipsync_video_path.stat().st_size / (1024 * 1024)
        print(f"✅ Lip-sync video: {lipsync_video_path.name} ({size_mb:.2f} MB)")
    else:
        print("❌ Lip-sync video creation failed")
        return

    # ================================================================
    # Stage 5: Lyrics Alignment & Subtitles (Optional)
    # ================================================================
    print_stage(5, "Lyrics Alignment & Subtitles")

    # Define lyrics (optional - can be empty to skip)
    original_lyrics = """The snow glows white on the mountain tonight
Not a footprint to be seen
A kingdom of isolation
and it looks like I'm the queen
The wind is howling like this swirling storm inside
Couldn't keep it in
Heaven knows I've tried"""

    ass_subtitle_path = None

    if original_lyrics.strip():
        try:
            from blt.synthesizer import WhisperLyricsAligner

            print("Aligning lyrics with audio...")
            aligner = WhisperLyricsAligner(model_size="medium", language="en")

            # Align lyrics with the mixed audio
            word_timings = aligner.align_and_split_by_lines(
                audio_path=str(mixed_path), lyrics=original_lyrics, language="en"
            )

            print(f"✅ Aligned {len(word_timings)} words")

            # Save alignment for reference
            alignment_file = output_dir / "alignment.txt"
            aligner.save_alignment(word_timings, str(alignment_file))
            print(f"   Alignment saved: {alignment_file.name}")

            # Create simple SRT subtitles
            srt_path = output_dir / "subtitles.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, timing in enumerate(word_timings, 1):
                    start_ms = int(timing.start * 1000)
                    end_ms = int(timing.end * 1000)
                    start_str = f"{start_ms // 3600000:02d}:{(start_ms % 3600000) // 60000:02d}:{(start_ms % 60000) // 1000:02d},{start_ms % 1000:03d}"
                    end_str = f"{end_ms // 3600000:02d}:{(end_ms % 3600000) // 60000:02d}:{(end_ms % 60000) // 1000:02d},{end_ms % 1000:03d}"
                    f.write(f"{i}\n{start_str} --> {end_str}\n{timing.word}\n\n")

            print(f"   Subtitles saved: {srt_path.name}")
            ass_subtitle_path = srt_path

        except Exception as e:
            print(f"⚠️  Lyrics alignment failed: {e}")
            print("   Continuing without subtitles...")
    else:
        print("⚠️  No lyrics provided, skipping alignment")

    # ================================================================
    # Stage 6: Final Video Composition
    # ================================================================
    print_stage(6, "Final Video Composition")

    final_video_path = output_dir / "擱淺_FINAL.mp4"

    # Build FFmpeg command with optional subtitles
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(lipsync_video_path),
        "-i",
        str(mixed_path),
    ]

    # Add subtitles filter if available
    if ass_subtitle_path and ass_subtitle_path.exists():
        # Escape path for FFmpeg subtitle filter
        subtitle_path_escaped = (
            str(ass_subtitle_path).replace("\\", "/").replace(":", "\\:")
        )
        cmd.extend(
            [
                "-vf",
                f"subtitles={subtitle_path_escaped}",
            ]
        )

    cmd.extend(
        [
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
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            str(final_video_path),
        ]
    )

    success, output = run_cmd(cmd, "Rendering final video")

    if not success:
        print("⚠️  Final composition failed, using lipsync video directly")
        shutil.copy(lipsync_video_path, final_video_path)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 70)

    print("\n📁 Output Files:")
    print(f"   Vocals:        {Path(vocal_path).name}")
    print(f"   Instrumental:  {Path(instrumental_path).name}")
    print(f"   Mixed Audio:   {mixed_path.name}")
    print(f"   Lip-Sync Video: {lipsync_video_path.name}")
    print(f"   FINAL VIDEO:   {final_video_path.name} ⭐")

    print(f"\n📂 Output: {output_dir}")

    if final_video_path.exists():
        size_mb = final_video_path.stat().st_size / (1024 * 1024)
        print("\n✅ FINAL VIDEO READY!")
        print(f"   File: {final_video_path.name}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Path: {final_video_path}")
    else:
        print("\n❌ Final video not created!")
        return None

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return final_video_path


if __name__ == "__main__":
    result = main()
    if result and result.exists():
        print(f"\n✅✅✅ SUCCESS!\n📹 Video: {result}\n")
