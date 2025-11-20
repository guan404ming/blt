#!/usr/bin/env python3
"""Script to create a JSON mapping of existing audio files to their MusicCaps captions."""

import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, ClapModel

# Constants
DATASET_NAME = "google/MusicCaps"
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = CACHE_DIR / "audio"
OUTPUT_FILE = CACHE_DIR / "audio_captions.json"
EMBEDDINGS_FILE = CACHE_DIR / "audio_captions_embeddings.pt"


def create_audio_caption_mapping(
    dataset,
    audio_dir: Path,
    output_file: Path,
) -> list[dict]:
    """Create a JSON mapping of existing audio files to their captions.

    Args:
        dataset: HuggingFace MusicCaps dataset
        audio_dir: Directory containing downloaded audio files
        output_file: Path to save the JSON mapping

    Returns:
        List of dictionaries with file and description keys
    """
    # Get all existing audio files
    existing_files = {f.name: f for f in audio_dir.glob("*.wav")}
    print(f"Found {len(existing_files)} audio files in {audio_dir}")

    # Build mapping from filename to caption
    train_data = dataset["train"]
    mappings = []

    for i in range(len(train_data)):
        example = train_data[i]
        ytid = example["ytid"]
        start_s = int(example["start_s"])
        end_s = int(example["end_s"])
        caption = example["caption"]

        filename = f"{ytid}_{start_s}_{end_s}.wav"

        if filename in existing_files:
            mappings.append({
                "file": filename,
                "description": caption,
            })

    # Sort by filename for consistency
    mappings.sort(key=lambda x: x["file"])

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=4, ensure_ascii=False)

    print(f"Created mapping for {len(mappings)} audio files")
    print(f"Saved to {output_file}")

    return mappings


def compute_and_save_embeddings(
    mappings: list[dict],
    output_file: Path,
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute CLAP text embeddings for all captions and save to disk.

    Args:
        mappings: List of dictionaries with file and description keys
        output_file: Path to save the embeddings tensor
        batch_size: Batch size for processing descriptions

    Returns:
        Tensor of shape (num_captions, embedding_dim) with normalized embeddings
    """
    if not mappings:
        print("No mappings to compute embeddings for")
        return torch.tensor([])

    print(f"\nComputing CLAP embeddings for {len(mappings)} captions...")

    # Load CLAP model
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

    # Get all descriptions
    descriptions = [item["description"] for item in mappings]

    # Process in batches
    all_embeddings = []
    num_batches = (len(descriptions) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Computing embeddings"):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(descriptions))
        batch_descriptions = descriptions[batch_start:batch_end]

        # Process batch
        inputs = processor(text=batch_descriptions, return_tensors="pt", padding=True)

        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)

        all_embeddings.append(embeddings)

    # Concatenate all batches
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Normalize embeddings
    all_embeddings = all_embeddings / all_embeddings.norm(dim=-1, keepdim=True)

    # Save to disk
    torch.save(all_embeddings, output_file)
    print(f"Saved embeddings to {output_file}")
    print(f"Embeddings shape: {all_embeddings.shape}")

    return all_embeddings


def main():
    """Load dataset and create audio-caption mapping."""
    print(f"Loading dataset: {DATASET_NAME}")
    print(f"Audio directory: {AUDIO_DIR}")

    # Load the dataset
    dataset = load_dataset(DATASET_NAME, cache_dir=str(CACHE_DIR))

    print(f"Dataset contains {len(dataset['train'])} entries")

    # Create the mapping
    mappings = create_audio_caption_mapping(dataset, AUDIO_DIR, OUTPUT_FILE)

    # Compute and save embeddings
    embeddings = compute_and_save_embeddings(mappings, EMBEDDINGS_FILE)

    # Show a few examples
    if mappings:
        print("\nSample entries:")
        print("=" * 50)
        for entry in mappings[:3]:
            print(f"\nFile: {entry['file']}")
            desc = entry['description']
            if len(desc) > 100:
                desc = desc[:100] + "..."
            print(f"Description: {desc}")

    return mappings, embeddings


if __name__ == "__main__":
    mappings = main()
