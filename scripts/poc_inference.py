#!/usr/bin/env python3
"""Proof of Concept: ICL Music Generation Inference.

This script demonstrates the basic workflow of:
1. Loading MusicGen with memory optimizations
2. Encoding audio examples
3. Generating music with ICL context

Optimized for RTX 5070 Ti (16GB VRAM).
"""

from pathlib import Path

import torch
import torchaudio


def check_gpu():
    """Check GPU availability and memory."""
    if not torch.cuda.is_available():
        print("CUDA not available. Please check your GPU setup.")
        return False

    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3

    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {total_memory:.1f} GB")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    return True


def run_poc():
    """Run the proof of concept inference."""
    print("=" * 60)
    print("Audio ICL - Proof of Concept")
    print("=" * 60)

    # Check GPU
    if not check_gpu():
        return

    print("\n[1/4] Loading MusicGen model...")
    print("      (This may take a minute on first run)")

    try:
        from audiocraft.models import MusicGen

        # Load medium model - good balance for 16GB VRAM
        model = MusicGen.get_pretrained("facebook/musicgen-medium")
        model.set_generation_params(
            use_sampling=True,
            top_k=250,
            duration=5,  # Short for testing
        )

        print("\n[2/4] Model loaded successfully!")

        # Check current VRAM usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"      VRAM allocated: {allocated:.2f} GB")
            print(f"      VRAM reserved: {reserved:.2f} GB")

        print("\n[3/4] Testing generation...")

        # Simple generation test
        prompt = "A calm acoustic guitar melody with soft percussion"

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model.generate([prompt])

        print(f"      Generated audio shape: {output.shape}")
        print(f"      Sample rate: {model.sample_rate} Hz")

        # Save output
        output_path = Path(__file__).parent.parent / "outputs"
        output_path.mkdir(exist_ok=True)

        audio_file = output_path / "poc_output.wav"
        torchaudio.save(
            str(audio_file),
            output[0].cpu(),
            sample_rate=model.sample_rate,
        )
        print(f"      Saved to: {audio_file}")

        print("\n[4/4] ICL Example Preparation (Demo)")

        # Demonstrate audio encoding
        print("      Testing audio encoding with EnCodec...")

        # Create a simple test audio
        test_audio = torch.randn(1, 1, model.sample_rate * 3)  # 3 seconds
        test_audio = test_audio.to(model.device)

        with torch.no_grad():
            encoded = model.compression_model.encode(test_audio)
            codes = encoded[0]
            print(f"      Encoded shape: {codes.shape}")
            print(f"      (batch, num_codebooks, seq_len)")

        print("\n      ICL workflow:")
        print("      1. Encode example audio -> tokens")
        print("      2. Pair with text description")
        print("      3. Concatenate: [Text₁][Audio₁]<SEP>[Text₂][Audio₂]<SEP>[Prompt]")
        print("      4. Generate with context")

    except ImportError as e:
        print(f"\n      Import error: {e}")
        print("      audiocraft not installed. Testing basic torch functionality...")

        # Fallback: just test basic CUDA/torch
        print("\n[2/4] Testing basic PyTorch CUDA...")
        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x.T
        print(f"      Matrix multiplication: {x.shape} @ {x.shape} = {y.shape}")
        print("      Basic CUDA test passed!")

        print("\n[3/4] Skipped (audiocraft not available)")
        print("\n[4/4] Skipped (audiocraft not available)")

    print("\n" + "=" * 60)
    print("PoC Complete!")
    print("=" * 60)

    # Final VRAM report
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak VRAM usage: {max_allocated:.2f} GB")
        print(f"Current VRAM usage: {allocated:.2f} GB")


if __name__ == "__main__":
    run_poc()
