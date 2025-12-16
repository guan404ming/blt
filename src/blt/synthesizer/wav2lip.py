"""Wav2Lip component for lip-sync video generation.

This module provides functionality to generate lip-synced videos using Wav2Lip,
which syncs facial movements with audio.
"""

import subprocess
from pathlib import Path
from typing import Optional


class Wav2Lip:
    """Generates lip-synced videos using Wav2Lip model.

    Wav2Lip is a state-of-the-art model for generating accurate lip-syncs
    in real-world talking face videos. It takes a video and audio, and generates
    a new video where the mouth movements are synced to the audio.

    Args:
        checkpoint_path: Path to the Wav2Lip model checkpoint (gan model)
        device: Device to run the model on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/wav2lip_gan.pth",
        device: Optional[str] = None,
    ):
        """Initialize Wav2Lip component.

        Args:
            checkpoint_path: Path to the Wav2Lip model checkpoint
            device: Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        """
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if self._has_cuda() else "cpu")

        print("Initializing Wav2Lip component")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Device: {self.device}")

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def generate_lipsync_video(
        self,
        face_video_path: str,
        audio_path: str,
        output_video_path: str,
        resize_factor: int = 1,
        pad: int = 0,
        pads: Optional[list] = None,
        fps: Optional[int] = None,
        mel_step_size: int = 16,
    ) -> str:
        """Generate a lip-synced video using Wav2Lip.

        Takes a video with a face and an audio file, and produces a new video
        where the mouth movements are synced to the audio.

        Args:
            face_video_path: Path to input video containing the face
            audio_path: Path to audio file to sync with
            output_video_path: Path to save the output lip-synced video
            resize_factor: Factor to resize the face detection. Higher = faster but less accurate
            pad: Padding to add around the detected face (deprecated, use pads)
            pads: List of [top, bottom, left, right] padding in pixels
            fps: Output video fps. If not specified, uses input video fps
            mel_step_size: Mel spectrogram step size for synchronization

        Returns:
            Path to the generated lip-synced video

        Raises:
            FileNotFoundError: If input files don't exist or checkpoint not found
            RuntimeError: If Wav2Lip inference fails
        """
        face_video_path = Path(face_video_path)
        audio_path = Path(audio_path)
        output_video_path = Path(output_video_path)

        # Validate input files
        if not face_video_path.exists():
            raise FileNotFoundError(f"Face video not found: {face_video_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Create output directory
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("WAV2LIP LIP-SYNC VIDEO GENERATION")
        print("=" * 60)
        print(f"Input video: {face_video_path}")
        print(f"Input audio: {audio_path}")
        print(f"Output video: {output_video_path}")

        try:
            # Import here to avoid import errors if Wav2Lip is not installed
            from wav2lip_inference import generate_lipsync

            # Prepare parameters
            kwargs = {
                "face_video": str(face_video_path),
                "audio": str(audio_path),
                "outfile": str(output_video_path),
                "checkpoint_path": self.checkpoint_path,
                "resize_factor": resize_factor,
                "mel_step_size": mel_step_size,
            }

            if pad > 0:
                kwargs["pad"] = pad
            if pads is not None:
                kwargs["pads"] = pads
            if fps is not None:
                kwargs["fps"] = fps

            print("\n🎬 Generating lip-sync video...")
            generate_lipsync(**kwargs)

            if output_video_path.exists():
                print("\n✓ Lip-sync video generated successfully")
                print(f"  Output: {output_video_path}")
                return str(output_video_path)
            else:
                raise RuntimeError("Output video was not created")

        except ImportError:
            print(
                "\n⚠️  Wav2Lip not installed. Attempting alternative inference method..."
            )
            return self._inference_via_cli(
                face_video_path,
                audio_path,
                output_video_path,
                resize_factor,
                pads or [pad] * 4 if pad > 0 else None,
                fps,
                mel_step_size,
            )

    def _inference_via_cli(
        self,
        face_video_path: Path,
        audio_path: Path,
        output_video_path: Path,
        resize_factor: int = 1,
        pads: Optional[list] = None,
        fps: Optional[int] = None,
        mel_step_size: int = 16,
    ) -> str:
        """Run Wav2Lip inference via command line.

        This is an alternative method when Wav2Lip Python module is not available.
        Requires Wav2Lip to be installed and inference.py to be in PATH.

        Args:
            face_video_path: Path to input video
            audio_path: Path to audio file
            output_video_path: Path to save output video
            resize_factor: Face detection resize factor
            pads: Padding [top, bottom, left, right]
            fps: Output frame rate
            mel_step_size: Mel spectrogram step size

        Returns:
            Path to the generated video

        Raises:
            RuntimeError: If inference fails
        """
        cmd = [
            "python",
            "inference.py",
            "--checkpoint_path",
            self.checkpoint_path,
            "--face",
            str(face_video_path),
            "--audio",
            str(audio_path),
            "--outfile",
            str(output_video_path),
            "--resize_factor",
            str(resize_factor),
            "--mel_step_size",
            str(mel_step_size),
        ]

        if pads is not None and len(pads) == 4:
            cmd.extend(["--pads"] + [str(p) for p in pads])

        if fps is not None:
            cmd.extend(["--fps", str(fps)])

        print("\n🎬 Running Wav2Lip inference via CLI...")
        print(f"  Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)

            if output_video_path.exists():
                print("\n✓ Lip-sync video generated successfully")
                print(f"  Output: {output_video_path}")
                return str(output_video_path)
            else:
                raise RuntimeError("Output video was not created")

        except subprocess.CalledProcessError as e:
            print("\n❌ Wav2Lip inference failed:")
            print(f"  Return code: {e.returncode}")
            print(f"  Stdout: {e.stdout}")
            print(f"  Stderr: {e.stderr}")
            raise RuntimeError(f"Wav2Lip inference failed: {e.stderr}")

    def batch_generate(
        self,
        face_videos: list,
        audio_files: list,
        output_dir: str,
        **kwargs,
    ) -> list:
        """Generate lip-synced videos in batch.

        Args:
            face_videos: List of face video paths
            audio_files: List of audio file paths
            output_dir: Directory to save output videos
            **kwargs: Additional arguments to pass to generate_lipsync_video

        Returns:
            List of paths to generated videos
        """
        if len(face_videos) != len(audio_files):
            raise ValueError(
                f"Number of videos ({len(face_videos)}) must match number of audios ({len(audio_files)})"
            )

        output_paths = []
        for i, (video, audio) in enumerate(zip(face_videos, audio_files)):
            video_name = Path(video).stem
            audio_name = Path(audio).stem
            output_path = Path(output_dir) / f"{video_name}_{audio_name}_lipsync.mp4"

            print(f"\n[{i + 1}/{len(face_videos)}] Processing...")
            output = self.generate_lipsync_video(
                str(video),
                str(audio),
                str(output_path),
                **kwargs,
            )
            output_paths.append(output)

        return output_paths
