# 🥪 BLT - Better Lyrics Translation Toolkit

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-apache-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

**BLT** is a toolkit for song translation and voice synthesis. The toolkit contains three modular components that can be used independently or combined through pre-defined pipelines.

## Toolkit Components

### 1. Translator
IPA-based lyrics translation tools with music constraints:

| Tool | Description |
|------|-------------|
| `LyricsTranslator` | Lyrics translation with syllable/rhyme preservation |
| `FeatureExtractor` | Extract music constraints (syllables, rhymes) from lyrics |
| `ConstraintValidator` | Validate translated lyrics against music constraints |

### 2. Synthesizer
TTS, alignment, and voice synthesis tools:

| Tool | Description |
|------|-------------|
| `VocalSeparator` | Separate vocals from instrumental tracks |
| `LyricsAligner` | Align lyrics timing with audio |
| `VoiceSynthesizer` | Synthesize vocals with new lyrics |

### 3. Pipeline
Pre-defined combinations of tools:

| Pipeline | Description |
|----------|-------------|
| `CoverSongPipeline` | End-to-end pipeline for generating translated cover songs |

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
