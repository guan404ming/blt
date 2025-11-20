"""
Proof of Concept script for MusicGen inference to compare CLAP scores
with and without in-context examples.
"""

import argparse
import csv
import os
import sys
import torch
import soundfile as sf
import json
from datetime import datetime
from pathlib import Path

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
AUDIO_CAPTIONS_PATH = PROJECT_ROOT / "data" / "audio_captions.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "audio_captions_embeddings.pt"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
AUDIO_CAPTIONS_PATH = PROJECT_ROOT / "data" / "audio_captions.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "audio_captions_embeddings.pt"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"


def get_clap_score(text, audio_file):
    """Calculates the CLAP score between a text prompt and an audio file."""
    original_argv = sys.argv
    sys.argv = [original_argv[0]]

    import torchaudio.transforms as T
    from transformers import AutoProcessor, ClapModel

    def load_audio_with_soundfile(file_path):
        """Loads an audio file with soundfile and converts it to a torch tensor."""
        waveform, sample_rate = sf.read(file_path, dtype="float32")
        # The model expects a mono channel audio, so we average channels if it's stereo
        if waveform.ndim > 1 and waveform.shape[1] > 1:
            waveform = waveform.mean(axis=1)
        return torch.from_numpy(waveform).unsqueeze(0), sample_rate

    # Load the model and processor
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

    # Load and process audio
    waveform, sample_rate = load_audio_with_soundfile(audio_file)

    # Resample if necessary
    if sample_rate != 48000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=48000)
        waveform = resampler(waveform)
        sample_rate = 48000

    # The processor expects a list of numpy arrays
    numpy_waveform = waveform.squeeze(0).numpy()

    # Process text and audio
    inputs = processor(
        text=text,
        audios=[numpy_waveform],
        return_tensors="pt",
        padding=True,
        sampling_rate=sample_rate,
    )

    # Get text and audio features
    with torch.no_grad():
        outputs = model(**inputs)
        text_features = outputs.text_embeds
        audio_features = outputs.audio_embeds

    # Normalize embeddings
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

    # Calculate score as dot product
    score = torch.matmul(text_features, audio_features.T)

    sys.argv = original_argv
    return score.item()


def run_generate_music(*args, **kwargs):
    """Wrapper for generate_music to isolate argument parsing."""
    original_argv = sys.argv
    sys.argv = [original_argv[0]]

    from omg import generate_music

    generate_music(*args, **kwargs)

    sys.argv = original_argv


def get_top_k_examples(
    prompt: str, k: int = 3, threshold: float = 0.7
) -> list[tuple[str, str]]:
    """Retrieves top-k examples from audio_captions.json based on cosine similarity.

    Uses pre-computed embeddings for fast retrieval.

    Args:
        prompt: The text prompt to match against descriptions
        k: Number of top examples to return
        threshold: Minimum similarity score to include an example (default: 0.7)

    Returns:
        List of tuples (description, audio_path) for the top-k most similar examples
    """
    original_argv = sys.argv
    sys.argv = [original_argv[0]]

    from transformers import AutoProcessor, ClapModel

    # Load captions data
    if not AUDIO_CAPTIONS_PATH.exists():
        print(
            f"Error: {AUDIO_CAPTIONS_PATH} not found. Run create_audio_caption_mapping.py first."
        )
        sys.argv = original_argv
        return []

    with open(AUDIO_CAPTIONS_PATH) as f:
        captions_data = json.load(f)

    if not captions_data:
        print("No captions found in audio_captions.json")
        sys.argv = original_argv
        return []

    # Load pre-computed embeddings if available
    if EMBEDDINGS_PATH.exists():
        print("Loading pre-computed embeddings...")
        desc_embeds = torch.load(EMBEDDINGS_PATH, weights_only=True)
    else:
        print(f"Warning: Pre-computed embeddings not found at {EMBEDDINGS_PATH}")
        print("Computing embeddings on-the-fly (this will be slow)...")
        print("Run create_audio_caption_mapping.py to pre-compute embeddings.")

        # Fall back to computing embeddings
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        descriptions = [item["description"] for item in captions_data]
        desc_inputs = processor(text=descriptions, return_tensors="pt", padding=True)

        with torch.no_grad():
            desc_embeds = model.get_text_features(**desc_inputs)
        desc_embeds = desc_embeds / desc_embeds.norm(dim=-1, keepdim=True)

    # Load CLAP model for prompt embedding only
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

    # Process prompt
    prompt_inputs = processor(text=[prompt], return_tensors="pt", padding=True)

    with torch.no_grad():
        prompt_embeds = model.get_text_features(**prompt_inputs)

    # Normalize prompt embedding
    prompt_embeds = prompt_embeds / prompt_embeds.norm(dim=-1, keepdim=True)

    # Calculate cosine similarities
    similarities = torch.matmul(prompt_embeds, desc_embeds.T).squeeze(0)

    # Get top-k indices
    top_k_indices = torch.topk(
        similarities, min(k, len(captions_data))
    ).indices.tolist()

    # Build result list
    examples = []
    print(f"\nTop-{k} similar examples for prompt: '{prompt}'")
    print(f"Threshold: {threshold}")
    print("=" * 60)

    for i, idx in enumerate(top_k_indices):
        item = captions_data[idx]
        audio_path = AUDIO_DIR / item["file"]
        similarity = similarities[idx].item()

        # Skip examples below threshold
        if similarity < threshold:
            print(f"{i + 1}. Score: {similarity:.4f} (below threshold, skipped)")
            continue

        if audio_path.exists():
            examples.append((item["description"], str(audio_path)))
            desc_preview = (
                item["description"][:80] + "..."
                if len(item["description"]) > 80
                else item["description"]
            )
            print(f"{i + 1}. Score: {similarity:.4f}")
            print(f"   File: {item['file']}")
            print(f"   Desc: {desc_preview}")
        else:
            print(f"Warning: Audio file not found: {audio_path}")

    print("=" * 60)
    print(f"Found {len(examples)} examples above threshold {threshold}")

    sys.argv = original_argv
    return examples


def get_top_k_examples(
    prompt: str, k: int = 3, threshold: float = 0.7
) -> list[tuple[str, str]]:
    """Retrieves top-k examples from audio_captions.json based on cosine similarity.

    Uses pre-computed embeddings for fast retrieval.

    Args:
        prompt: The text prompt to match against descriptions
        k: Number of top examples to return
        threshold: Minimum similarity score to include an example (default: 0.7)

    Returns:
        List of tuples (description, audio_path) for the top-k most similar examples
    """
    original_argv = sys.argv
    sys.argv = [original_argv[0]]

    from transformers import AutoProcessor, ClapModel

    # Load captions data
    if not AUDIO_CAPTIONS_PATH.exists():
        print(
            f"Error: {AUDIO_CAPTIONS_PATH} not found. Run create_audio_caption_mapping.py first."
        )
        sys.argv = original_argv
        return []

    with open(AUDIO_CAPTIONS_PATH) as f:
        captions_data = json.load(f)

    if not captions_data:
        print("No captions found in audio_captions.json")
        sys.argv = original_argv
        return []

    # Load pre-computed embeddings if available
    if EMBEDDINGS_PATH.exists():
        print("Loading pre-computed embeddings...")
        desc_embeds = torch.load(EMBEDDINGS_PATH, weights_only=True)
    else:
        print(f"Warning: Pre-computed embeddings not found at {EMBEDDINGS_PATH}")
        print("Computing embeddings on-the-fly (this will be slow)...")
        print("Run create_audio_caption_mapping.py to pre-compute embeddings.")

        # Fall back to computing embeddings
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        descriptions = [item["description"] for item in captions_data]
        desc_inputs = processor(text=descriptions, return_tensors="pt", padding=True)

        with torch.no_grad():
            desc_embeds = model.get_text_features(**desc_inputs)
        desc_embeds = desc_embeds / desc_embeds.norm(dim=-1, keepdim=True)

    # Load CLAP model for prompt embedding only
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

    # Process prompt
    prompt_inputs = processor(text=[prompt], return_tensors="pt", padding=True)

    with torch.no_grad():
        prompt_embeds = model.get_text_features(**prompt_inputs)

    # Normalize prompt embedding
    prompt_embeds = prompt_embeds / prompt_embeds.norm(dim=-1, keepdim=True)

    # Calculate cosine similarities
    similarities = torch.matmul(prompt_embeds, desc_embeds.T).squeeze(0)

    # Get top-k indices
    top_k_indices = torch.topk(
        similarities, min(k, len(captions_data))
    ).indices.tolist()

    # Build result list
    examples = []
    print(f"\nTop-{k} similar examples for prompt: '{prompt}'")
    print(f"Threshold: {threshold}")
    print("=" * 60)

    for i, idx in enumerate(top_k_indices):
        item = captions_data[idx]
        audio_path = AUDIO_DIR / item["file"]
        similarity = similarities[idx].item()

        # Skip examples below threshold
        if similarity < threshold:
            print(f"{i + 1}. Score: {similarity:.4f} (below threshold, skipped)")
            continue

        if audio_path.exists():
            examples.append((item["description"], str(audio_path)))
            desc_preview = (
                item["description"][:80] + "..."
                if len(item["description"]) > 80
                else item["description"]
            )
            print(f"{i + 1}. Score: {similarity:.4f}")
            print(f"   File: {item['file']}")
            print(f"   Desc: {desc_preview}")
        else:
            print(f"Warning: Audio file not found: {audio_path}")

    print("=" * 60)
    print(f"Found {len(examples)} examples above threshold {threshold}")

    sys.argv = original_argv
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate music and calculate CLAP score."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="jazz with only saxophone and keyboard interplay",
        help="Text prompt for music generation.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to a text file with multiple prompts (one per line).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=20,
        help="Duration of the generated music in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save the generated audio files and scores.",
    )
    parser.add_argument(
        "--examples-json",
        type=str,
        default=None,
        help="Path to a JSON file with descriptions and audio file names for the examples. If not provided, uses automatic retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top similar examples to retrieve for ICL (default: 3).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum similarity score for an example to be included (default: 0.7).",
        default=None,
        help="Path to a JSON file with descriptions and audio file names for the examples. If not provided, uses automatic retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top similar examples to retrieve for ICL (default: 3).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum similarity score for an example to be included (default: 0.7).",
    )
    parser.add_argument(
        "--no-icl",
        action="store_true",
        help="Do not run in-context learning generation.",
    )

    args = parser.parse_args()

    # Get list of prompts
    prompts = []
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = [args.prompt]

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CSV file for results
    csv_path = output_dir / "scores.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "prompt_id",
            "prompt",
            "baseline_score",
            "icl_score",
            "num_examples",
            "baseline_audio",
            "icl_audio",
        ]
    )

    print(f"\nResults will be saved to: {output_dir}")
    print(f"Scores will be saved to: {csv_path}")
    print("=" * 60)

    # Process each prompt
    for idx, prompt in enumerate(prompts):
        prompt_id = f"{idx + 1:03d}"
        print(f"\n[{prompt_id}/{len(prompts):03d}] Processing: {prompt[:50]}...")

        # Define output paths
        baseline_output_path = output_dir / f"{prompt_id}_baseline.wav"
        icl_output_path = output_dir / f"{prompt_id}_icl.wav"

        # 1. Generate baseline audio
        print("Generating baseline audio...")
        run_generate_music(
            prompt=prompt,
            duration=args.duration,
            output_path=str(baseline_output_path),
        )
        print(f"Baseline audio generated at {baseline_output_path}")

        # 2. Calculate CLAP score for baseline
        print("Calculating CLAP score for baseline...")
        baseline_score = get_clap_score(prompt, str(baseline_output_path))
        print(f"CLAP score for baseline: {baseline_score:.4f}")

        icl_score = None
        num_examples = 0

        # 3. Generate and score in-context learning audio
        if not args.no_icl:
            print("\nGenerating in-context learning audio...")

            examples = []

            if args.examples_json:
                # Use provided JSON file
                json_path = Path(args.examples_json)
                examples_dir = json_path.parent

                if not json_path.exists():
                    print(
                        f"Warning: JSON file not found at {json_path}. Skipping ICL generation."
                    )
                else:
                    with open(json_path) as f:
                        example_data = json.load(f)

                    for item in example_data:
                        audio_path = examples_dir / item["file"]
                        if audio_path.exists():
                            examples.append((item["description"], str(audio_path)))
                        else:
                            print(
                                f"Warning: {audio_path} not found for example '{item['description']}'"
                            )
            else:
                # Use automatic retrieval based on cosine similarity
                print(
                    f"Retrieving top-{args.top_k} examples based on cosine similarity..."
                )
                examples = get_top_k_examples(
                    prompt, k=args.top_k, threshold=args.threshold
                )

            num_examples = len(examples)

            if not examples:
                print("No valid in-context examples found. Skipping ICL generation.")
            else:
                run_generate_music(
                    prompt=prompt,
                    duration=args.duration,
                    output_path=str(icl_output_path),
                    examples=examples,
                )
                print(f"In-context learning audio generated at {icl_output_path}")

                print("Calculating CLAP score for in-context learning...")
                icl_score = get_clap_score(prompt, str(icl_output_path))
                print(f"CLAP score for in-context learning: {icl_score:.4f}")

                print("\nComparison:")
                print(f"  - Baseline CLAP score: {baseline_score:.4f}")
                print(f"  - ICL CLAP score:      {icl_score:.4f}")

        # Write to CSV
        csv_writer.writerow(
            [
                prompt_id,
                prompt,
                f"{baseline_score:.4f}",
                f"{icl_score:.4f}" if icl_score is not None else "",
                num_examples,
                baseline_output_path.name,
                icl_output_path.name if icl_score is not None else "",
            ]
        )
        csv_file.flush()

    csv_file.close()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total prompts processed: {len(prompts)}")
    print(f"Results saved to: {output_dir}")
    print(f"Scores saved to: {csv_path}")
    print("=" * 60)
