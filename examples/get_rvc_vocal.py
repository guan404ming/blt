import argparse
import os
from blt.synthesizer.rvc_converter import RvcConverter


def main():
    parser = argparse.ArgumentParser(description="RVC vocal conversion for audio files")
    parser.add_argument(
        "--input",
        default="examples/vocal-擱淺.wav",
        help="Vocal reference wav file (default: 擱淺_vocal.wav)",
    )
    parser.add_argument(
        "--output", default="examples/", help="Output .mp3 or .wav file"
    )
    parser.add_argument(
        "--model",
        help="RVC model (.pth) file (default: RVC_model/model.pth)",
    )
    parser.add_argument(
        "--index",
        help="RVC index (.index) file (default: RVC_model/model.index)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}")
        return
    if not os.path.isfile(args.input):
        print(f"Vocal reference file not found: {args.input}")
        return
    if not os.path.isfile(args.model):
        print(f"Model file not found: {args.model}")
        return
    if not os.path.isfile(args.index):
        print(f"Index file not found: {args.index}")
        return

    try:
        converter = RvcConverter()
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
