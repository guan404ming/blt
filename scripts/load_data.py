#!/usr/bin/env python3
"""Script to load MusicCaps dataset from Hugging Face and download audio from YouTube."""

import subprocess
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Constants
DATASET_NAME = "google/MusicCaps"
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = CACHE_DIR / "audio"
AUDIO_FORMAT = "wav"
MAX_SAMPLES = None  # Set to an integer to limit downloads, None for all


def download_audio_clip(
    ytid: str,
    start_s: float,
    end_s: float,
    output_dir: Path,
    audio_format: str = "wav",
) -> Path | None:
    """Download a specific audio clip from YouTube.

    Args:
        ytid: YouTube video ID
        start_s: Start time in seconds
        end_s: End time in seconds
        output_dir: Directory to save the audio file
        audio_format: Audio format (default: wav)

    Returns:
        Path to the downloaded file, or None if download failed
    """
    output_file = output_dir / f"{ytid}_{int(start_s)}_{int(end_s)}.{audio_format}"

    # Skip if already downloaded
    if output_file.exists():
        return output_file

    url = f"https://www.youtube.com/watch?v={ytid}"

    # Build yt-dlp command
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format",
        audio_format,
        "--audio-quality",
        "0",  # Best quality
        "-o",
        str(output_dir / f"{ytid}_full.%(ext)s"),
        "--download-sections",
        f"*{start_s}-{end_s}",
        "--force-keyframes-at-cuts",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # yt-dlp with --download-sections names the file differently
        # Find the downloaded file and rename it
        possible_files = list(output_dir.glob(f"{ytid}_full*.{audio_format}"))
        if possible_files:
            possible_files[0].rename(output_file)
            return output_file
    except subprocess.CalledProcessError:
        pass  # Video might be unavailable

    return None


def download_musiccaps_audio(
    dataset,
    output_dir: Path,
    max_samples: int | None = None,
    audio_format: str = "wav",
) -> dict:
    """Download audio clips for MusicCaps dataset.

    Args:
        dataset: HuggingFace dataset with ytid, start_s, end_s columns
        output_dir: Directory to save audio files
        max_samples: Maximum number of samples to download (None for all)
        audio_format: Audio format (default: wav)

    Returns:
        Dictionary with download statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = dataset["train"]
    n_samples = (
        len(train_data) if max_samples is None else min(max_samples, len(train_data))
    )

    stats = {"success": 0, "failed": 0, "skipped": 0}

    print(f"\nDownloading {n_samples} audio clips to {output_dir}")

    for i in tqdm(range(n_samples), desc="Downloading"):
        example = train_data[i]
        ytid = example["ytid"]
        start_s = example["start_s"]
        end_s = example["end_s"]

        output_file = output_dir / f"{ytid}_{int(start_s)}_{int(end_s)}.{audio_format}"

        if output_file.exists():
            stats["skipped"] += 1
            continue

        result = download_audio_clip(ytid, start_s, end_s, output_dir, audio_format)

        if result:
            stats["success"] += 1
        else:
            stats["failed"] += 1

    return stats


def main():
    """Load dataset and download audio."""
    # Create directories if they don't exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATASET_NAME}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Audio directory: {AUDIO_DIR}")

    # Load the dataset
    dataset = load_dataset(DATASET_NAME, cache_dir=str(CACHE_DIR))

    print(f"\nDataset structure: {dataset}")
    print(f"\nNumber of examples: {len(dataset['train'])}")

    # Display column names and types
    print(f"\nColumns: {dataset['train'].column_names}")
    print(f"\nFeatures: {dataset['train'].features}")

    # Show a few example entries
    print("\n" + "=" * 50)
    print("Sample entries:")
    print("=" * 50)

    for i in range(min(3, len(dataset["train"]))):
        example = dataset["train"][i]
        print(f"\n--- Example {i + 1} ---")
        for key, value in example.items():
            # Truncate long text for display
            if isinstance(value, str) and len(value) > 200:
                value = value[:200] + "..."
            print(f"{key}: {value}")

    # Check if yt-dlp is installed
    try:
        subprocess.run(
            ["yt-dlp", "--version"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nError: yt-dlp is not installed. Install it with:")
        print("  uv pip install yt-dlp")
        sys.exit(1)

    # Download audio
    stats = download_musiccaps_audio(
        dataset,
        AUDIO_DIR,
        max_samples=MAX_SAMPLES,
        audio_format=AUDIO_FORMAT,
    )

    print("\n" + "=" * 50)
    print("Download Statistics:")
    print("=" * 50)
    print(f"Successfully downloaded: {stats['success']}")
    print(f"Already existed (skipped): {stats['skipped']}")
    print(f"Failed (unavailable): {stats['failed']}")

    return dataset


if __name__ == "__main__":
    dataset = main()
