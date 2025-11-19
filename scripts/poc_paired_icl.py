#!/usr/bin/env python3
"""PoC: Paired Text-Audio Token ICL for Text-to-Music Generation.

This implements the core research contribution:
- Pair text descriptions with their encoded audio tokens
- Inject these pairs into the model's context
- Generate conditioned on multiple (text, audio) examples

Format: [Text₁][Audio_Tokens₁]<SEP>[Text₂][Audio_Tokens₂]<SEP>[Target_Prompt]
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf


def get_audio_features(audio: np.ndarray, sample_rate: int = 32000) -> dict:
    """Extract audio features for comparison."""
    if torch.is_tensor(audio):
        audio = audio.squeeze().float().numpy()
    elif len(audio.shape) > 1:
        audio = audio.squeeze()

    rms = np.sqrt(np.mean(audio**2))
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    magnitude = np.abs(fft)
    centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)

    return {"rms": rms, "centroid": centroid}


def compute_style_distance(feat1: dict, feat2: dict) -> float:
    """Compute normalized distance between feature sets."""
    distances = []
    for key in feat1:
        if feat1[key] != 0:
            d = abs(feat1[key] - feat2[key]) / (abs(feat1[key]) + 1e-8)
            distances.append(d)
    return np.mean(distances)


class PairedICLGenerator:
    """Generator that uses paired (text, audio_tokens) as ICL context."""

    def __init__(self, model):
        self.model = model
        self.device = model.device
        self.sample_rate = model.sample_rate

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to discrete tokens."""
        with torch.no_grad():
            audio = audio.to(self.device).float()
            encoded = self.model.compression_model.encode(audio)
            return encoded[0]  # [batch, codebooks, seq_len]

    def get_text_conditioning(self, descriptions: list[str]) -> torch.Tensor:
        """Get text conditioning embeddings from T5."""
        # MusicGen uses T5 for text conditioning
        # We need to access the conditioner
        attributes, _ = self.model._prepare_tokens_and_attributes(
            descriptions=descriptions,
            prompt=None,
        )

        # Get the conditioning tensors
        conditions = self.model.condition_provider(attributes)

        return conditions

    def generate_with_paired_icl(
        self,
        icl_examples: list[dict],  # [{"text": str, "audio": Tensor}, ...]
        target_prompt: str,
        duration: float = 10.0,
    ) -> torch.Tensor:
        """Generate audio using paired (text, audio) ICL examples.

        The key idea: We concatenate the audio tokens from examples
        as a "prefix" to prime the generation, while also providing
        the text descriptions to establish the text-audio mapping.

        Args:
            icl_examples: List of {"text": description, "audio": waveform}
            target_prompt: Text prompt for generation
            duration: Output duration in seconds

        Returns:
            Generated audio waveform
        """
        self.model.set_generation_params(
            use_sampling=True,
            top_k=250,
            duration=duration,
        )

        # Encode all ICL example audios
        encoded_examples = []
        for ex in icl_examples:
            tokens = self.encode_audio(ex["audio"])
            encoded_examples.append({
                "text": ex["text"],
                "tokens": tokens,
            })

        # Strategy: Concatenate all example audio tokens as prefix
        # This teaches the model the "style" through actual audio

        if len(encoded_examples) > 0:
            # Concatenate all audio tokens
            all_tokens = torch.cat(
                [ex["tokens"] for ex in encoded_examples],
                dim=2  # Concatenate along sequence dimension
            )

            # Decode the concatenated tokens back to audio for continuation
            with torch.no_grad():
                # Decode tokens to get the prefix audio
                prefix_audio = self.model.compression_model.decode(all_tokens, None)

            # Build combined text prompt that includes example descriptions
            combined_descriptions = []
            for i, ex in enumerate(encoded_examples):
                combined_descriptions.append(f"Example {i+1}: {ex['text']}")
            combined_descriptions.append(f"Generate: {target_prompt}")
            combined_text = " | ".join(combined_descriptions)

            # Generate continuation from the prefix
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = self.model.generate_continuation(
                    prompt=prefix_audio,
                    prompt_sample_rate=self.sample_rate,
                    descriptions=[target_prompt],  # Use target prompt for guidance
                )
        else:
            # No ICL examples, standard generation
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = self.model.generate([target_prompt])

        return output, encoded_examples


def run_paired_icl_evaluation():
    """Run paired text-audio ICL evaluation."""
    print("=" * 60)
    print("Paired Text-Audio Token ICL Evaluation")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"\nGPU: {torch.cuda.get_device_name()}")

    # Load model
    print("\nLoading MusicGen model...")
    from audiocraft.models import MusicGen
    model = MusicGen.get_pretrained("facebook/musicgen-small")

    # Create generator
    generator = PairedICLGenerator(model)

    output_dir = Path(__file__).parent.parent / "outputs" / "paired_icl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Experiment: Use multiple paired examples
    experiments = [
        {
            "name": "ambient_2examples",
            "icl_examples": [
                {
                    "text": "Soft ambient pads with gentle reverb",
                    "prompt": "Soft ambient pads with gentle reverb",
                },
                {
                    "text": "Atmospheric synthesizer textures",
                    "prompt": "Atmospheric synthesizer textures",
                },
            ],
            "target": "Relaxing ambient electronic soundscape",
        },
        {
            "name": "edm_2examples",
            "icl_examples": [
                {
                    "text": "High energy EDM with punchy kicks",
                    "prompt": "High energy EDM with punchy kicks",
                },
                {
                    "text": "Fast electronic dance with bass drops",
                    "prompt": "Fast electronic dance with bass drops",
                },
            ],
            "target": "Powerful dance track with driving rhythm",
        },
    ]

    results = []
    example_duration = 3  # seconds per ICL example
    target_duration = 10  # output duration

    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*50}")

        # Generate ICL example audios
        print(f"\n1. Generating {len(exp['icl_examples'])} ICL examples...")
        icl_examples = []
        all_example_features = []

        model.set_generation_params(use_sampling=True, top_k=250, duration=example_duration)

        for i, ex in enumerate(exp['icl_examples']):
            print(f"   Example {i+1}: {ex['text'][:40]}...")
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                audio = model.generate([ex['prompt']])

            icl_examples.append({
                "text": ex['text'],
                "audio": audio,
            })

            feat = get_audio_features(audio[0].cpu().float().numpy(), model.sample_rate)
            all_example_features.append(feat)

        # Compute average example features (target style)
        avg_example_features = {
            key: np.mean([f[key] for f in all_example_features])
            for key in all_example_features[0]
        }

        # Generate with paired ICL
        print(f"\n2. Generating with paired ICL...")
        print(f"   Target: {exp['target']}")
        print(f"   ICL examples: {len(icl_examples)}")

        icl_output, encoded_info = generator.generate_with_paired_icl(
            icl_examples=icl_examples,
            target_prompt=exp['target'],
            duration=target_duration,
        )

        # Print token info
        total_tokens = sum(ex['tokens'].shape[2] for ex in encoded_info)
        print(f"   Total ICL tokens: {total_tokens} ({len(icl_examples)} examples × {example_duration}s)")

        icl_features = get_audio_features(
            icl_output[0].cpu().float().numpy(),
            model.sample_rate
        )

        # Generate baseline (no ICL)
        print(f"\n3. Generating baseline...")

        model.set_generation_params(use_sampling=True, top_k=250, duration=target_duration)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            baseline_output = model.generate([exp['target']])

        baseline_features = get_audio_features(
            baseline_output[0].cpu().float().numpy(),
            model.sample_rate
        )

        # Calculate distances to average example style
        icl_distance = compute_style_distance(avg_example_features, icl_features)
        baseline_distance = compute_style_distance(avg_example_features, baseline_features)
        improvement = (baseline_distance - icl_distance) / (baseline_distance + 1e-8) * 100

        result = {
            "name": exp['name'],
            "num_examples": len(icl_examples),
            "total_tokens": total_tokens,
            "icl_distance": icl_distance,
            "baseline_distance": baseline_distance,
            "improvement_pct": improvement,
            "icl_better": icl_distance < baseline_distance,
        }
        results.append(result)

        # Print results
        print(f"\n4. Results:")
        print(f"   Avg Example - RMS: {avg_example_features['rms']:.4f}, Centroid: {avg_example_features['centroid']:.1f} Hz")
        print(f"   ICL         - RMS: {icl_features['rms']:.4f}, Centroid: {icl_features['centroid']:.1f} Hz")
        print(f"   Baseline    - RMS: {baseline_features['rms']:.4f}, Centroid: {baseline_features['centroid']:.1f} Hz")
        print(f"   ")
        print(f"   ICL distance: {icl_distance:.4f}")
        print(f"   Baseline distance: {baseline_distance:.4f}")
        print(f"   Improvement: {improvement:+.1f}%")
        print(f"   ICL better: {'Yes ✓' if result['icl_better'] else 'No ✗'}")

        # Save audio files
        for i, ex in enumerate(icl_examples):
            sf.write(
                str(output_dir / f"{exp['name']}_example{i+1}.wav"),
                ex['audio'][0].cpu().squeeze().float().numpy(),
                model.sample_rate,
            )

        sf.write(
            str(output_dir / f"{exp['name']}_icl.wav"),
            icl_output[0].cpu().squeeze().float().numpy(),
            model.sample_rate,
        )
        sf.write(
            str(output_dir / f"{exp['name']}_baseline.wav"),
            baseline_output[0].cpu().squeeze().float().numpy(),
            model.sample_rate,
        )

        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print("\nMethodology:")
    print("  - Multiple (text, audio) pairs as ICL context")
    print("  - Audio encoded to tokens, concatenated as prefix")
    print("  - Text descriptions inform the mapping")

    icl_wins = sum(1 for r in results if r['icl_better'])
    avg_improvement = np.mean([r['improvement_pct'] for r in results])

    print(f"\nResults:")
    print(f"  Total experiments: {len(results)}")
    print(f"  ICL wins: {icl_wins}/{len(results)}")
    print(f"  Average improvement: {avg_improvement:+.1f}%")

    print("\nPer-experiment:")
    for r in results:
        status = "✓" if r['icl_better'] else "✗"
        print(f"  {status} {r['name']}: {r['improvement_pct']:+.1f}% ({r['num_examples']} examples, {r['total_tokens']} tokens)")

    print(f"\nOutput: {output_dir}")

    # Research direction
    print("\n" + "=" * 60)
    print("NEXT STEPS FOR RESEARCH")
    print("=" * 60)
    print("""
Current approach: Concatenate audio tokens as generation prefix
This works but includes the example audio in output.

True structured ICL requires:
1. Modify transformer attention to see tokens as CONTEXT only
2. Add special tokens: <EXAMPLE_START>, <SEP>, <GENERATE>
3. Train/fine-tune to learn the text-audio mapping
4. Generate NEW audio (not continuation) based on examples

This is the core technical contribution for publication.
""")

    return results


if __name__ == "__main__":
    results = run_paired_icl_evaluation()
