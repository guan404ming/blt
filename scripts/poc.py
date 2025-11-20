"""
Proof of Concept script for MusicGen inference to compare CLAP scores
with and without in-context examples.
"""

import argparse
import os
import sys
import torch
import soundfile as sf
import json
from pathlib import Path


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
        "--duration",
        type=int,
        default=20,
        help="Duration of the generated music in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/examples",
        help="Directory to save the generated audio files.",
    )
    parser.add_argument(
        "--examples-json",
        type=str,
        default="scripts/examples/jazz.json",
        help="Path to a JSON file with descriptions and audio file names for the examples.",
    )
    parser.add_argument(
        "--no-icl",
        action="store_true",
        help="Do not run in-context learning generation.",
    )

    args = parser.parse_args()

    # Define output paths
    baseline_output_path = os.path.join(args.output_dir, "POC_baseline.wav")
    icl_output_path = os.path.join(args.output_dir, "POC_icl.wav")

    # 1. Generate baseline audio
    print("Generating baseline audio...")
    run_generate_music(
        prompt=args.prompt,
        duration=args.duration,
        output_path=baseline_output_path,
    )
    print(f"Baseline audio generated at {baseline_output_path}")

    # 2. Calculate CLAP score for baseline
    print("Calculating CLAP score for baseline...")
    baseline_score = get_clap_score(args.prompt, baseline_output_path)
    print(f"CLAP score for baseline: {baseline_score:.4f}")

    # 3. Generate and score in-context learning audio
    if not args.no_icl:
        print("\nGenerating in-context learning audio...")

        examples = []
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

            if not examples:
                print(
                    "No valid in-context examples found from the JSON file. Skipping ICL generation."
                )
            else:
                run_generate_music(
                    prompt=args.prompt,
                    duration=args.duration,
                    output_path=icl_output_path,
                    examples=examples,
                )
                print(f"In-context learning audio generated at {icl_output_path}")

                print("Calculating CLAP score for in-context learning...")
                icl_score = get_clap_score(args.prompt, icl_output_path)
                print(f"CLAP score for in-context learning: {icl_score:.4f}")

                print("\nComparison:")
                print(f"  - Baseline CLAP score: {baseline_score:.4f}")
                print(f"  - ICL CLAP score:      {icl_score:.4f}")
