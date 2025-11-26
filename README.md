# 🥪 BLT - Better Lyrics Translation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-apache-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

**BLT** is an AI-powered song translation tool that allows you to translate any song into another language. It features a smart lyrics translation engine that rewrites lyrics to fit the song's melody, rhythm, and rhyme scheme, and then synthesizes the vocals to match the original singer's voice.

## Features

- **BLT (Better Lyrics Translation)**: Automatically translates and adapts lyrics to target languages, ensuring they fit the original melody's syllable count and rhyme constraints.
- **Voice Cloning**: Preserves the unique timbre and style of the original artist in the translated version.
- **Multilingual Support**: Translate songs between any language supported by modern LLMs.

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

## Usage

To generate a translated cover song, use the `examples/gen_translated_song.py` script:

```bash
uv run python examples/gen_translated_song.py \
    --audio "path/to/your/song.mp3" \
    --old-lyrics-file "path/to/original/lyrics.txt" \
    --new-lyrics-file "path/to/new/lyrics.txt" \
    --output-name "my_cover_song"
```

**Parameters:**

- `--audio`: Path to the original song audio file.
- `--old-lyrics-file`: Path to a text file containing the original lyrics.
- `--new-lyrics-file`: Path to a text file containing the new lyrics.
- `--output-name`: The name for the generated cover song files.
- `--device`: The device to run the models on (`cuda` or `cpu`).

## Acknowledgments

- [Demucs](https://github.com/facebookresearch/demucs) by Facebook Research
- [XTTS](https://github.com/coqui-ai/TTS) by Coqui AI

## License

Apache License 2.0
