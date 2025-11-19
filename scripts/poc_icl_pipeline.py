#!/usr/bin/env python3
"""PoC: Complete ICL Pipeline for Text-to-Music Generation.

This script demonstrates:
1. Loading MusicCaps dataset (text-audio pairs)
2. In-context learning with audio examples
3. Generating music with ICL
4. Simple evaluation metrics

Optimized for RTX 5070 Ti (16GB VRAM).
"""

import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm


def check_gpu():
    """Check GPU availability and memory."""
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return False

    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3

    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {total_memory:.1f} GB")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    return True


def download_musiccaps_sample(num_samples: int = 10):
    """Download a sample of MusicCaps dataset.

    MusicCaps contains YouTube audio with text descriptions.
    We'll use a subset for the PoC.
    """
    print(f"\n[1/5] Loading MusicCaps dataset ({num_samples} samples)...")

    # Load MusicCaps from HuggingFace
    dataset = load_dataset("google/MusicCaps", split="train")

    # Sample random examples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    samples = [dataset[i] for i in indices]

    print(f"      Loaded {len(samples)} samples")
    print(f"      Example caption: '{samples[0]['caption'][:100]}...'")

    return samples


def create_synthetic_examples(model, num_examples: int = 3):
    """Create synthetic audio examples for ICL demo.

    Since downloading YouTube audio requires additional setup,
    we'll generate synthetic examples using the model itself.
    """
    print(f"\n[2/5] Creating {num_examples} synthetic ICL examples...")

    examples = []
    prompts = [
        "Upbeat electronic dance music with synthesizer and drums",
        "Soft piano melody with ambient background",
        "Acoustic folk song with guitar strumming",
        "Jazz trio with saxophone, bass, and drums",
        "Cinematic orchestral music with strings and brass",
    ]

    for i, prompt in enumerate(prompts[:num_examples]):
        print(f"      Generating example {i+1}/{num_examples}: {prompt[:40]}...")

        with torch.no_grad():
            audio = model.generate([prompt])

        examples.append({
            "text": prompt,
            "audio": audio[0].cpu(),
        })

        # Clear cache after each generation
        torch.cuda.empty_cache()

    print(f"      Created {len(examples)} examples")
    return examples


def encode_icl_examples(model, examples: list) -> list:
    """Encode audio examples to tokens for ICL.

    This converts audio waveforms to discrete EnCodec tokens
    that can be used as context for generation.
    """
    print("\n[3/5] Encoding ICL examples to audio tokens...")

    encoded_examples = []
    for i, ex in enumerate(examples):
        audio = ex["audio"].unsqueeze(0).to(model.device)

        with torch.no_grad():
            # Encode to discrete tokens
            encoded = model.compression_model.encode(audio)
            codes = encoded[0]  # (batch, num_codebooks, seq_len)

        encoded_examples.append({
            "text": ex["text"],
            "codes": codes.cpu(),
            "shape": codes.shape,
        })

        print(f"      Example {i+1}: {codes.shape} tokens")

    return encoded_examples


def generate_with_icl(
    model,
    target_prompt: str,
    icl_examples: list,
    duration: float = 5.0,
) -> torch.Tensor:
    """Generate music using ICL examples as context.

    This is a simplified ICL approach for PoC:
    - Concatenate example descriptions with target prompt
    - Use audio continuation for style transfer

    Note: Full ICL implementation would require model modifications
    to properly inject encoded examples into the generation loop.
    """
    print(f"\n[4/5] Generating with ICL context...")
    print(f"      Target: {target_prompt}")
    print(f"      Using {len(icl_examples)} ICL examples")

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration,
    )

    if icl_examples:
        # PoC ICL approach: Use audio continuation
        # Start generation from the last ICL example's audio
        # This creates a style transfer effect

        # Get the last example's audio as continuation prompt
        prompt_audio = icl_examples[-1]["audio"].unsqueeze(0).to(model.device)

        # Truncate to 3 seconds for the prompt
        prompt_samples = int(3 * model.sample_rate)
        if prompt_audio.shape[-1] > prompt_samples:
            prompt_audio = prompt_audio[..., :prompt_samples]

        # Build a combined prompt that references the ICL examples
        _ = f"Similar style to: {icl_examples[0]['text'][:50]}. Generate: {target_prompt}"

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Generate continuation from the prompt audio
            output = model.generate_continuation(
                prompt=prompt_audio,
                prompt_sample_rate=model.sample_rate,
                descriptions=[target_prompt],  # Use original prompt for clarity
            )
    else:
        # Standard generation without ICL
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = model.generate([target_prompt])

    print(f"      Generated shape: {output.shape}")
    return output


def simple_evaluation(
    generated_audio: torch.Tensor,
    reference_audio: torch.Tensor | None = None,
    sample_rate: int = 32000,
) -> dict:
    """Simple evaluation metrics for generated audio.

    Metrics:
    - Duration check
    - Energy/loudness
    - Spectral centroid (brightness)
    - Zero-crossing rate (noisiness)
    - Spectral bandwidth
    - Spectral rolloff
    """
    print("\n[5/5] Evaluating generated audio...")

    # Convert to float32 for numpy compatibility
    audio = generated_audio.squeeze().float().numpy()

    # Duration
    duration = len(audio) / sample_rate

    # RMS Energy (loudness)
    rms = np.sqrt(np.mean(audio**2))

    # Zero-crossing rate
    zcr = np.mean(np.abs(np.diff(np.signbit(audio))))

    # Simple spectral analysis
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    magnitude = np.abs(fft)

    # Spectral centroid
    spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)

    # Spectral bandwidth (spread around centroid)
    spectral_bandwidth = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-8)
    )

    # Spectral rolloff (freq below which 85% of energy is contained)
    cumsum = np.cumsum(magnitude)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    spectral_rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    metrics = {
        "duration_sec": duration,
        "rms_energy": float(rms),
        "zero_crossing_rate": float(zcr),
        "spectral_centroid_hz": float(spectral_centroid),
        "spectral_bandwidth_hz": float(spectral_bandwidth),
        "spectral_rolloff_hz": float(spectral_rolloff),
    }

    print(f"      Duration: {duration:.2f}s")
    print(f"      RMS Energy: {rms:.4f}")
    print(f"      Zero-crossing rate: {zcr:.4f}")
    print(f"      Spectral centroid: {spectral_centroid:.1f} Hz")
    print(f"      Spectral bandwidth: {spectral_bandwidth:.1f} Hz")
    print(f"      Spectral rolloff: {spectral_rolloff:.1f} Hz")

    # Style consistency with reference
    if reference_audio is not None:
        ref = reference_audio.squeeze().float().numpy()
        ref_fft = np.fft.rfft(ref)
        ref_magnitude = np.abs(ref_fft)
        ref_centroid = np.sum(freqs[:len(ref_magnitude)] * ref_magnitude) / (np.sum(ref_magnitude) + 1e-8)

        style_similarity = 1.0 - abs(spectral_centroid - ref_centroid) / max(spectral_centroid, ref_centroid)
        metrics["style_similarity"] = float(style_similarity)
        print(f"      Style similarity to reference: {style_similarity:.2%}")

    return metrics


def compute_style_transfer_score(
    icl_output: torch.Tensor,
    baseline_output: torch.Tensor,
    icl_example: torch.Tensor,
    sample_rate: int = 32000,
) -> dict:
    """Compute how well the ICL output captures the style of the example.

    This measures whether ICL successfully transferred style characteristics
    from the example to the generated output.
    """
    def get_spectral_features(audio):
        audio = audio.squeeze().float().numpy()
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
        magnitude = np.abs(fft)

        centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
        bandwidth = np.sqrt(
            np.sum(((freqs - centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-8)
        )
        rms = np.sqrt(np.mean(audio**2))

        return {"centroid": centroid, "bandwidth": bandwidth, "rms": rms}

    icl_feat = get_spectral_features(icl_output)
    baseline_feat = get_spectral_features(baseline_output)
    example_feat = get_spectral_features(icl_example)

    # Distance from example
    icl_dist = sum([
        abs(icl_feat[k] - example_feat[k]) / (example_feat[k] + 1e-8)
        for k in icl_feat
    ]) / len(icl_feat)

    baseline_dist = sum([
        abs(baseline_feat[k] - example_feat[k]) / (example_feat[k] + 1e-8)
        for k in baseline_feat
    ]) / len(baseline_feat)

    # ICL should be closer to example than baseline
    improvement = (baseline_dist - icl_dist) / (baseline_dist + 1e-8)

    return {
        "icl_distance_to_example": float(icl_dist),
        "baseline_distance_to_example": float(baseline_dist),
        "style_transfer_improvement": float(improvement),
        "icl_better": icl_dist < baseline_dist,
    }


def run_icl_poc():
    """Run the complete ICL PoC pipeline."""
    print("=" * 60)
    print("Audio ICL - Complete PoC Pipeline")
    print("=" * 60)

    # Check GPU
    if not check_gpu():
        return

    # Create output directory
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Load model
    print("\nLoading MusicGen model...")
    from audiocraft.models import MusicGen

    # Use small model for faster iteration on PoC
    model = MusicGen.get_pretrained("facebook/musicgen-small")

    print(f"Model loaded on: {model.device}")
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM allocated: {allocated:.2f} GB")

    # Step 1: Load dataset info (we'll use synthetic data for PoC)
    _ = download_musiccaps_sample(num_samples=5)

    # Step 2: Create synthetic ICL examples
    icl_examples = create_synthetic_examples(model, num_examples=2)

    # Step 3: Encode examples
    encoded_examples = encode_icl_examples(model, icl_examples)

    # Step 4: Generate with ICL
    target_prompt = "A relaxing ambient piece with soft pads and gentle melody"

    # Generate with ICL (using melody conditioning as proxy)
    generated_icl = generate_with_icl(
        model,
        target_prompt,
        icl_examples,
        duration=5.0,
    )

    # Also generate without ICL for comparison
    print("\n      Generating baseline (no ICL)...")
    model.set_generation_params(use_sampling=True, top_k=250, duration=5.0)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        generated_baseline = model.generate([target_prompt])

    # Step 5: Evaluate
    print("\n" + "-" * 40)
    print("ICL Generation Metrics:")
    metrics_icl = simple_evaluation(generated_icl[0].cpu(), sample_rate=model.sample_rate)

    print("\nBaseline Generation Metrics:")
    metrics_baseline = simple_evaluation(generated_baseline[0].cpu(), sample_rate=model.sample_rate)

    # Compute style transfer score
    print("\n" + "-" * 40)
    print("Style Transfer Analysis:")
    style_score = compute_style_transfer_score(
        generated_icl[0].cpu(),
        generated_baseline[0].cpu(),
        icl_examples[-1]["audio"],  # The example used for continuation
        model.sample_rate,
    )

    print(f"      ICL distance to example: {style_score['icl_distance_to_example']:.4f}")
    print(f"      Baseline distance to example: {style_score['baseline_distance_to_example']:.4f}")
    print(f"      Style transfer improvement: {style_score['style_transfer_improvement']:.2%}")
    print(f"      ICL is better: {style_score['icl_better']}")

    # Save outputs using soundfile (more compatible)
    import soundfile as sf

    sf.write(
        str(output_dir / "poc_icl_output.wav"),
        generated_icl[0].cpu().squeeze().float().numpy(),
        model.sample_rate,
    )
    sf.write(
        str(output_dir / "poc_baseline_output.wav"),
        generated_baseline[0].cpu().squeeze().float().numpy(),
        model.sample_rate,
    )

    # Save ICL examples
    for i, ex in enumerate(icl_examples):
        sf.write(
            str(output_dir / f"poc_icl_example_{i}.wav"),
            ex["audio"].squeeze().float().numpy(),
            model.sample_rate,
        )

    print("\n" + "=" * 60)
    print("PoC Complete!")
    print("=" * 60)

    print(f"\nOutputs saved to: {output_dir}")
    print("  - poc_icl_output.wav      : ICL generation")
    print("  - poc_baseline_output.wav : Baseline generation")
    print("  - poc_icl_example_*.wav   : ICL examples")

    # Final VRAM report
    if torch.cuda.is_available():
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak VRAM usage: {max_allocated:.2f} GB")

    return {
        "icl_metrics": metrics_icl,
        "baseline_metrics": metrics_baseline,
        "encoded_examples": [
            {"text": e["text"], "shape": e["shape"]}
            for e in encoded_examples
        ],
    }


if __name__ == "__main__":
    results = run_icl_poc()

    if results:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print("\nICL Examples used:")
        for ex in results["encoded_examples"]:
            print(f"  - {ex['text'][:50]}... ({ex['shape']})")
