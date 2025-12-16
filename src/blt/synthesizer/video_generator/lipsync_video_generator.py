"""Lip-sync video generation using Wav2Lip.

This module provides functionality to generate lip-synced videos using Wav2Lip,
which syncs facial movements with audio.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional


class LipSyncedVideoGenerator:
    """Generates lip-synced videos using Wav2Lip model.

    Wav2Lip is a state-of-the-art model for generating accurate lip-syncs
    in real-world talking face videos. It takes a face image/video and audio,
    and generates a new video where the mouth movements are synced to the audio.

    Args:
        wav2lip_dir: Path to Wav2Lip directory (default: auto-detect)
        checkpoint_path: Path to the Wav2Lip model checkpoint (default: checkpoints/wav2lip_gan.pth)
        device: Device to run the model on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        wav2lip_dir: Optional[str] = None,
        checkpoint_path: str = "checkpoints/wav2lip_gan.pth",
        device: Optional[str] = None,
    ):
        """Initialize LipSyncedVideoGenerator component.

        Args:
            wav2lip_dir: Path to Wav2Lip directory. If None, auto-detects.
            checkpoint_path: Path to the Wav2Lip model checkpoint (relative to wav2lip_dir)
            device: Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        """
        # Auto-detect Wav2Lip directory if not provided
        if wav2lip_dir is None:
            module_dir = Path(__file__).parent.parent
            wav2lip_dir = module_dir / "Wav2Lip"

        self.wav2lip_dir = Path(wav2lip_dir)
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if self._has_cuda() else "cpu")

        print("Initializing LipSyncedVideoGenerator")
        print(f"  Wav2Lip directory: {self.wav2lip_dir}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Device: {self.device}")

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def generate(
        self,
        face_path: str,
        audio_path: str,
        output_path: str,
        resize_factor: int = 1,
    ) -> str:
        """Generate a lip-synced video using Wav2Lip.

        Takes a face image/video and an audio file, and produces a video
        where the mouth movements are synced to the audio.

        Args:
            face_path: Path to input image/video containing the face
            audio_path: Path to audio file to sync with
            output_path: Path to save the output lip-synced video
            resize_factor: Factor to resize the face detection. Higher = faster but less accurate (default: 1)

        Returns:
            Path to the generated lip-synced video

        Raises:
            FileNotFoundError: If input files don't exist or Wav2Lip not found
            RuntimeError: If Wav2Lip inference fails
        """
        face_path = Path(face_path).resolve()
        audio_path = Path(audio_path).resolve()
        output_path = Path(output_path).resolve()

        # Validate input files
        if not face_path.exists():
            raise FileNotFoundError(f"Face image/video not found: {face_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not self.wav2lip_dir.exists():
            raise FileNotFoundError(
                f"Wav2Lip directory not found: {self.wav2lip_dir}\n"
                "Please ensure Wav2Lip is installed in the synthesizer directory."
            )

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("LIP-SYNC VIDEO GENERATION")
        print("=" * 60)
        print(f"Face: {face_path}")
        print(f"Audio: {audio_path}")
        print(f"Output: {output_path}")
        print()

        # Run Wav2Lip inference
        return self._run_inference(face_path, audio_path, output_path, resize_factor)

    def _run_inference(
        self,
        face_path: Path,
        audio_path: Path,
        output_path: Path,
        resize_factor: int,
    ) -> str:
        """Run Wav2Lip inference via command line.

        Args:
            face_path: Path to input face image/video
            audio_path: Path to audio file
            output_path: Path to save output video
            resize_factor: Face detection resize factor

        Returns:
            Path to the generated video

        Raises:
            RuntimeError: If inference fails
        """
        import os

        # Save current directory
        original_cwd = Path.cwd()

        try:
            # Change to Wav2Lip directory
            os.chdir(self.wav2lip_dir)

            # Build command
            cmd = [
                sys.executable,
                "inference.py",
                "--checkpoint_path",
                self.checkpoint_path,
                "--face",
                str(face_path),
                "--audio",
                str(audio_path),
                "--outfile",
                str(output_path),
                "--resize_factor",
                str(resize_factor),
            ]

            print("🎬 Running Wav2Lip inference...")
            print(
                f"   Command: python inference.py --checkpoint_path {self.checkpoint_path}"
            )
            print(f"            --face {face_path}")
            print(f"            --audio {audio_path}")
            print(f"            --outfile {output_path}")
            print()

            # Run command
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )

            # Return to original directory
            os.chdir(original_cwd)

            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print()
                print("=" * 60)
                print("✓ SUCCESS!")
                print("=" * 60)
                print(f"Video created: {output_path}")
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
            print("Error output:")
            print(e.stderr)
            if e.stdout:
                print("\nStandard output:")
                print(e.stdout)
            raise RuntimeError(f"Wav2Lip inference failed: {e.stderr}")

        except subprocess.TimeoutExpired:
            os.chdir(original_cwd)
            raise RuntimeError("Wav2Lip took too long (>10 minutes)")

        except Exception as e:
            os.chdir(original_cwd)
            raise RuntimeError(f"Wav2Lip inference failed: {e}")
