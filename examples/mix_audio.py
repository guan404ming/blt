"""Example: Mix two audio files together using AudioMixer.

This example demonstrates how to use the AudioMixer class to combine
vocals and instrumental tracks into a single mixed audio file.
"""

from blt.synthesizer import AudioMixer


def main():
    # Input audio files
    vocals_path = "assets/vocal-edited-擱淺.wav"
    instrumental_path = "assets/separated_audio/擱淺_instrumental.wav"

    # Output mixed audio file
    output_path = "assets/擱淺_mixed.wav"

    print("=" * 60)
    print("AUDIO MIXING EXAMPLE")
    print("=" * 60)
    print(f"Vocal track: {vocals_path}")
    print(f"Instrumental track: {instrumental_path}")
    print(f"Output: {output_path}")
    print()

    # Create mixer
    mixer = AudioMixer(normalize=True)

    # Mix the audio files
    # You can adjust volume levels (1.0 = 100%, 0.5 = 50%, etc.)
    result_path = mixer.mix(
        audio1_path=vocals_path,
        audio2_path=instrumental_path,
        output_path=output_path,
        volume1=1.0,  # Vocal volume
        volume2=0.8,  # Instrumental volume (slightly lower)
    )

    print()
    print("=" * 60)
    print("MIXING RESULTS")
    print("=" * 60)
    print(f"Mixed audio: {result_path}")
    print()
    print("✓ Mixing complete!")
    print()
    print("You can now use this mixed file for:")
    print("  - Final audio output")
    print("  - Video lip-sync with Wav2Lip")
    print("  - Further audio processing")


if __name__ == "__main__":
    main()
