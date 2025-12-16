import argparse
from blt.synthesizer import RetrievalBasedVoiceConverter


def main():
    parser = argparse.ArgumentParser(description="RVC vocal conversion for audio files")
    parser.add_argument(
        "--input",
        default="assets/vocal-擱淺.wav",
        help="Vocal reference wav file (default: 擱淺_vocal.wav)",
    )
    parser.add_argument("--output", default="assets/", help="Output .mp3 or .wav file")
    parser.add_argument(
        "--model",
        default="assets/model.pth",
        help="RVC model (.pth) file (default: assets/model.pth)",
    )
    parser.add_argument(
        "--index",
        default="assets/model.index",
        help="RVC index (.index) file (default: assets/model.index)",
    )
    args = parser.parse_args()

    try:
        converter = RetrievalBasedVoiceConverter()
        converter.run(
            audio_path=args.input,
            model_path=args.model,
            index_path=args.index,
            output_path=args.output,
        )
        print(f"RVC conversion complete: {args.output}")
    except Exception as e:
        print(f"Error during RVC conversion: {e}")


if __name__ == "__main__":
    main()
