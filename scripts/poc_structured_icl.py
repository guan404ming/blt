#!/usr/bin/env python3
"""PoC: True Structured ICL for Text-to-Music Generation.

This implements proper ICL where:
1. ICL uses TEXT PROMPTS enriched with example descriptions (fair comparison)
2. Both ICL and Baseline use the SAME generation method (model.generate)
3. ICL advantage comes from better prompt engineering with examples

Key insight: True ICL should work through the text conditioning pathway,
not through audio continuation which adds unfair extra information.
"""

from pathlib import Path

import numpy as np
import torch
import soundfile as sf


def get_audio_features(audio: np.ndarray, sample_rate: int = 32000) -> dict:
    """Extract audio features for comparison."""
    if torch.is_tensor(audio):
        audio = audio.squeeze().float().numpy()
    elif len(audio.shape) > 1:
        audio = audio.squeeze()

    # Time domain
    rms = np.sqrt(np.mean(audio**2))
    zcr = np.mean(np.abs(np.diff(np.signbit(audio))))

    # Frequency domain
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    magnitude = np.abs(fft)

    # Spectral features
    centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
    bandwidth = np.sqrt(
        np.sum(((freqs - centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-8)
    )

    return {
        "rms": rms,
        "centroid": centroid,
        "bandwidth": bandwidth,
    }


def compute_style_distance(feat1: dict, feat2: dict) -> float:
    """Compute normalized distance between two feature sets."""
    distances = []
    for key in feat1:
        if feat1[key] != 0:
            d = abs(feat1[key] - feat2[key]) / (abs(feat1[key]) + 1e-8)
            distances.append(d)
    return np.mean(distances)


def build_icl_prompt(examples: list[dict], target: str) -> str:
    """Build a structured ICL prompt from examples.

    This creates a prompt that includes example descriptions to guide
    the model's understanding of the desired style.

    Format:
    "Examples of the style:
    1. [example 1 description]
    2. [example 2 description]

    Now generate: [target description]"
    """
    prompt_parts = ["Examples of the desired style:"]

    for i, ex in enumerate(examples, 1):
        prompt_parts.append(f"{i}. {ex['description']}")

    prompt_parts.append("")
    prompt_parts.append(f"Now generate in the same style: {target}")

    return "\n".join(prompt_parts)


def run_structured_icl_evaluation():
    """Run ICL evaluation with consistent methodology."""
    print("=" * 60)
    print("Structured ICL Evaluation (Fair Comparison)")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load model
    print("\nLoading MusicGen model...")
    from audiocraft.models import MusicGen
    model = MusicGen.get_pretrained("facebook/musicgen-small")

    output_dir = Path(__file__).parent.parent / "outputs" / "structured_icl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Experiment design:
    # - Define style through multiple example descriptions
    # - ICL: Use structured prompt with examples
    # - Baseline: Use only target description
    # - Both use model.generate() - fair comparison

    experiments = [
        {
            "name": "ambient_electronic",
            "examples": [
                {"description": "Soft ambient electronic pads with gentle reverb and space"},
                {"description": "Atmospheric synthesizer textures with slow evolving sounds"},
                {"description": "Calm electronic ambient with warm bass and ethereal melodies"},
            ],
            "target": "Relaxing ambient electronic piece",
            "baseline_target": "Relaxing ambient electronic piece",
        },
        {
            "name": "energetic_edm",
            "examples": [
                {"description": "High energy EDM with punchy kicks and aggressive synth leads"},
                {"description": "Fast tempo electronic dance music with heavy bass drops"},
                {"description": "Energetic house beat with driving rhythm and powerful synths"},
            ],
            "target": "Powerful dance track with strong rhythm",
            "baseline_target": "Powerful dance track with strong rhythm",
        },
        {
            "name": "acoustic_folk",
            "examples": [
                {"description": "Gentle acoustic guitar fingerpicking with warm tone"},
                {"description": "Soft folk melody with delicate string harmonics"},
                {"description": "Peaceful acoustic piece with natural room ambience"},
            ],
            "target": "Calm acoustic guitar melody",
            "baseline_target": "Calm acoustic guitar melody",
        },
    ]

    results = []
    duration = 10  # seconds

    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*50}")

        # Build ICL prompt
        icl_prompt = build_icl_prompt(exp['examples'], exp['target'])
        baseline_prompt = exp['baseline_target']

        print(f"\nICL Prompt:\n{icl_prompt}")
        print(f"\nBaseline Prompt: {baseline_prompt}")

        # Generate with ICL prompt
        print(f"\n1. Generating with ICL prompt...")
        model.set_generation_params(use_sampling=True, top_k=250, duration=duration)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            icl_audio = model.generate([icl_prompt])

        icl_features = get_audio_features(
            icl_audio[0].cpu().float().numpy(),
            model.sample_rate
        )

        # Generate baseline
        print(f"2. Generating baseline...")

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            baseline_audio = model.generate([baseline_prompt])

        baseline_features = get_audio_features(
            baseline_audio[0].cpu().float().numpy(),
            model.sample_rate
        )

        # Generate "ground truth" - what the style should sound like
        # Use one of the example descriptions directly
        print(f"3. Generating reference (example style)...")

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            reference_audio = model.generate([exp['examples'][0]['description']])

        reference_features = get_audio_features(
            reference_audio[0].cpu().float().numpy(),
            model.sample_rate
        )

        # Compute distances to reference style
        icl_distance = compute_style_distance(reference_features, icl_features)
        baseline_distance = compute_style_distance(reference_features, baseline_features)

        improvement = (baseline_distance - icl_distance) / (baseline_distance + 1e-8) * 100

        result = {
            "name": exp['name'],
            "icl_distance": icl_distance,
            "baseline_distance": baseline_distance,
            "improvement_pct": improvement,
            "icl_better": icl_distance < baseline_distance,
            "features": {
                "reference": reference_features,
                "icl": icl_features,
                "baseline": baseline_features,
            }
        }
        results.append(result)

        # Print results
        print(f"\n4. Results:")
        print(f"   Reference - RMS: {reference_features['rms']:.4f}, Centroid: {reference_features['centroid']:.1f} Hz")
        print(f"   ICL       - RMS: {icl_features['rms']:.4f}, Centroid: {icl_features['centroid']:.1f} Hz")
        print(f"   Baseline  - RMS: {baseline_features['rms']:.4f}, Centroid: {baseline_features['centroid']:.1f} Hz")
        print(f"   ")
        print(f"   ICL distance to reference: {icl_distance:.4f}")
        print(f"   Baseline distance to reference: {baseline_distance:.4f}")
        print(f"   Improvement: {improvement:+.1f}%")
        print(f"   ICL better: {'Yes ✓' if result['icl_better'] else 'No ✗'}")

        # Save audio files
        sf.write(
            str(output_dir / f"{exp['name']}_reference.wav"),
            reference_audio[0].cpu().squeeze().float().numpy(),
            model.sample_rate,
        )
        sf.write(
            str(output_dir / f"{exp['name']}_icl.wav"),
            icl_audio[0].cpu().squeeze().float().numpy(),
            model.sample_rate,
        )
        sf.write(
            str(output_dir / f"{exp['name']}_baseline.wav"),
            baseline_audio[0].cpu().squeeze().float().numpy(),
            model.sample_rate,
        )

        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    icl_wins = sum(1 for r in results if r['icl_better'])
    avg_improvement = np.mean([r['improvement_pct'] for r in results])

    print(f"\nMethodology: Both ICL and Baseline use model.generate()")
    print(f"ICL advantage: Structured prompt with example descriptions")
    print(f"")
    print(f"Total experiments: {len(results)}")
    print(f"ICL wins: {icl_wins}/{len(results)}")
    print(f"Average improvement: {avg_improvement:+.1f}%")

    print("\nPer-experiment results:")
    for r in results:
        status = "✓" if r['icl_better'] else "✗"
        print(f"  {status} {r['name']}: {r['improvement_pct']:+.1f}%")

    print(f"\nOutput files saved to: {output_dir}")

    # Conclusion
    print("\n" + "=" * 60)
    if icl_wins > len(results) / 2:
        print("CONCLUSION: Structured ICL shows advantage!")
        print("Prompt engineering with example descriptions helps")
        print("the model better understand the target style.")
    elif icl_wins == len(results) / 2:
        print("CONCLUSION: Mixed results")
        print("Structured prompts show inconsistent improvement.")
        print("May need better prompt design or model fine-tuning.")
    else:
        print("CONCLUSION: Baseline performs better")
        print("Simple prompts may be more effective than verbose ones.")
        print("Consider: model may not benefit from long prompts.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_structured_icl_evaluation()
