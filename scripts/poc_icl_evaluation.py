#!/usr/bin/env python3
"""PoC: ICL Evaluation - Proving ICL Advantage.

This script demonstrates that ICL improves generation by:
1. Using matched example styles (same style as target)
2. Measuring style consistency between example and output
3. Comparing ICL vs baseline on multiple metrics

Key insight: ICL should make output MORE similar to the example style.
"""

from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm


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

    # Spectral flatness (how noise-like vs tonal)
    geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-8)))
    arithmetic_mean = np.mean(magnitude)
    flatness = geometric_mean / (arithmetic_mean + 1e-8)

    return {
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "flatness": flatness,
    }


def compute_style_distance(feat1: dict, feat2: dict) -> float:
    """Compute normalized distance between two feature sets."""
    distances = []
    for key in feat1:
        if feat1[key] != 0:
            d = abs(feat1[key] - feat2[key]) / (abs(feat1[key]) + 1e-8)
            distances.append(d)
    return np.mean(distances)


def run_evaluation():
    """Run ICL evaluation experiment."""
    print("=" * 60)
    print("ICL Evaluation: Proving Style Transfer Advantage")
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

    output_dir = Path(__file__).parent.parent / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Experiment: Style-matched ICL
    # We generate examples with specific styles, then use them as ICL context
    # to generate new content with the SAME target style

    styles = [
        {
            "name": "ambient",
            "example_prompt": "Soft ambient electronic music with gentle pads and reverb",
            "target_prompt": "Relaxing ambient soundscape with atmospheric textures",
            "expected": {"low_rms": True, "high_flatness": True},
        },
        {
            "name": "energetic",
            "example_prompt": "High energy electronic dance music with heavy bass and drums",
            "target_prompt": "Powerful EDM track with driving rhythm and bass drops",
            "expected": {"high_rms": True, "low_flatness": True},
        },
    ]

    results = []

    for style in styles:
        print(f"\n{'='*40}")
        print(f"Testing style: {style['name']}")
        print(f"{'='*40}")

        # 1. Generate style example
        print(f"\n1. Generating ICL example...")
        print(f"   Prompt: {style['example_prompt'][:50]}...")

        # Generate longer audio (30 seconds)
        model.set_generation_params(use_sampling=True, top_k=250, duration=30)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            example_audio = model.generate([style['example_prompt']])

        example_features = get_audio_features(
            example_audio[0].cpu().float().numpy(),
            model.sample_rate
        )

        # 2. Generate with ICL (continuation from example)
        print(f"\n2. Generating with ICL context...")
        print(f"   Target: {style['target_prompt'][:50]}...")

        prompt_audio = example_audio[0:1, :, :int(3 * model.sample_rate)]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            icl_audio = model.generate_continuation(
                prompt=prompt_audio,
                prompt_sample_rate=model.sample_rate,
                descriptions=[style['target_prompt']],
            )

        icl_features = get_audio_features(
            icl_audio[0].cpu().float().numpy(),
            model.sample_rate
        )

        # 3. Generate baseline (no ICL)
        print(f"\n3. Generating baseline (no ICL)...")

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            baseline_audio = model.generate([style['target_prompt']])

        baseline_features = get_audio_features(
            baseline_audio[0].cpu().float().numpy(),
            model.sample_rate
        )

        # 4. Compute distances
        icl_distance = compute_style_distance(example_features, icl_features)
        baseline_distance = compute_style_distance(example_features, baseline_features)

        improvement = (baseline_distance - icl_distance) / (baseline_distance + 1e-8) * 100

        # Store results
        result = {
            "style": style['name'],
            "example_features": example_features,
            "icl_features": icl_features,
            "baseline_features": baseline_features,
            "icl_distance": icl_distance,
            "baseline_distance": baseline_distance,
            "improvement_pct": improvement,
            "icl_better": icl_distance < baseline_distance,
        }
        results.append(result)

        # Print results
        print(f"\n4. Results for '{style['name']}':")
        print(f"   Example RMS: {example_features['rms']:.4f}, Centroid: {example_features['centroid']:.1f} Hz")
        print(f"   ICL RMS: {icl_features['rms']:.4f}, Centroid: {icl_features['centroid']:.1f} Hz")
        print(f"   Baseline RMS: {baseline_features['rms']:.4f}, Centroid: {baseline_features['centroid']:.1f} Hz")
        print(f"   ")
        print(f"   ICL distance to example: {icl_distance:.4f}")
        print(f"   Baseline distance to example: {baseline_distance:.4f}")
        print(f"   Improvement: {improvement:+.1f}%")
        print(f"   ICL better: {'Yes' if result['icl_better'] else 'No'}")

        # Save audio files
        sf.write(
            str(output_dir / f"{style['name']}_example.wav"),
            example_audio[0].cpu().squeeze().float().numpy(),
            model.sample_rate,
        )
        sf.write(
            str(output_dir / f"{style['name']}_icl.wav"),
            icl_audio[0].cpu().squeeze().float().numpy(),
            model.sample_rate,
        )
        sf.write(
            str(output_dir / f"{style['name']}_baseline.wav"),
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

    print(f"\nTotal styles tested: {len(results)}")
    print(f"ICL wins: {icl_wins}/{len(results)}")
    print(f"Average improvement: {avg_improvement:+.1f}%")

    print("\nPer-style results:")
    for r in results:
        status = "✓" if r['icl_better'] else "✗"
        print(f"  {status} {r['style']}: {r['improvement_pct']:+.1f}%")

    print(f"\nOutput files saved to: {output_dir}")

    # Final verdict
    print("\n" + "=" * 60)
    if icl_wins > len(results) / 2:
        print("CONCLUSION: ICL demonstrates style transfer advantage!")
        print("The continuation-based ICL successfully transfers style")
        print("characteristics from examples to generated outputs.")
    else:
        print("CONCLUSION: More sophisticated ICL needed")
        print("The simple continuation approach shows mixed results.")
        print("Full ICL implementation with proper context injection")
        print("would likely show better results.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_evaluation()
