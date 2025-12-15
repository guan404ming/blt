

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from blt.synthesizer.rvc_converter import RvcConverter

DEFAULT_VOCAL_PATH = os.path.join(os.path.dirname(__file__), '擱淺_vocal.wav')
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'RVC_model', 'model.pth')
DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'RVC_model', 'model.index')
DEFAULT_OUTPUT_PATH = os.path.dirname(__file__)

def main():
    parser = argparse.ArgumentParser(description='RVC vocal conversion for audio files')
    parser.add_argument('--input', default=DEFAULT_VOCAL_PATH, help='Vocal reference wav file (default: 擱淺_vocal.wav)')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_PATH,help='Output .mp3 or .wav file')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help='RVC model (.pth) file (default: RVC_model/model.pth)')
    parser.add_argument('--index', default=DEFAULT_INDEX_PATH, help='RVC index (.index) file (default: RVC_model/model.index)')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f'Input file not found: {args.input}')
        return
    if not os.path.isfile(args.input):
        print(f'Vocal reference file not found: {args.input}')
        return
    if not os.path.isfile(args.model):
        print(f'Model file not found: {args.model}')
        return
    if not os.path.isfile(args.index):
        print(f'Index file not found: {args.index}')
        return

    try:
        converter = RvcConverter()
        converter.run(
            audio_path=args.input,
            model_path=args.model,
            index_path=args.index,
            output_path=args.output
        )
        print(f'RVC conversion complete: {args.output}')
    except Exception as e:
        print(f'Error during RVC conversion: {e}')

if __name__ == '__main__':
    main()
