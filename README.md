# 🥪 BLT - Better Lyrics Translation Toolkit

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-apache-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

**BLT** is a toolkit for song translation and voice synthesis. The toolkit contains three modular components that can be used independently or combined through pre-defined pipelines.

## Toolkit Components

### 1. Translator

**IPA-based lyrics translation tools with music constraints:**

| Tool                  | Description                                          |
| --------------------- | ---------------------------------------------------- |
| `LyricsTranslator`    | Main translator with syllable/rhyme preservation     |
| `SoramimiTranslator`  | Phonetic (soramimi/空耳) translator - creates text that sounds like the original |

**Music Constraints Extracted:**

1. **syllable_counts**: `list[int]` (ex. [4, 3])

   - Chinese: Character-based
   - Other languages: IPA vowel nuclei

2. **syllable_patterns**: `list[list[int]]` (ex. [[1, 1, 2], [1, 2]])

   - **With audio (WIP)**: Alignment problem - timing sync with vocals
   - **Without audio**: Word segmentation problem
     - Chinese: HanLP tokenizer
     - English: Space splitting
     - Other languages: LLM-based

3. **rhyme_scheme**: `str` (ex. AB)
   - Chinese: Pinyin finals
   - Other languages: IPA phonemes

4. **ipa_similarity**: `float` (ex. 0.5)
   - Phonetic similarity threshold for soramimi translation
   - Measured using IPA phoneme matching between source and target

<details open>
<summary><b>Translation Flow</b></summary>

```mermaid
flowchart TD
    A[Source Lyrics] --> B[LyricsAnalyzer]
    B --> |Extract Constraints| C{LyricsTranslator}
    C --> |Generate Translation| D[ConstraintValidator]
    D --> |Check Constraints| E{Valid or Max Retries}
    E --> |No| C
    E --> |Yes| F[Target Lyrics]

    style B fill:#64b5f6,stroke:#1976d2,stroke-width:2px,color:#fff
    style C fill:#1976d2,stroke:#0d47a1,stroke-width:2px,color:#fff
    style D fill:#42a5f5,stroke:#1976d2,stroke-width:2px,color:#fff
```

</details>

### 2. (WIP) Synthesizer

TTS, alignment, and voice synthesis tools:

| Tool               | Description                              |
| ------------------ | ---------------------------------------- |
| `VocalSeparator`   | Separate vocals from instrumental tracks |
| `LyricsAligner`    | Align lyrics timing with audio           |
| `VoiceSynthesizer` | Synthesize vocals with new lyrics        |
| `RvcConverter`     | Convert voice using RVC models           |

### 3. (WIP) Pipeline

Pre-defined combinations of tools:

| Pipeline            | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `CoverSongPipeline` | End-to-end pipeline for generating translated cover songs |

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

## Usage

### Soramimi Translation (Phonetic Matching)

To generate soramimi (空耳) lyrics that sound like the original, use the `examples/get_soramimi_lyrics.py` script:

```bash
uv run python examples/get_soramimi_lyrics.py
```

**Parameters:**

- `-f, --file`: Path to source lyrics file (default: `data/lyrics-let-it-go.txt`)
- `-s, --source`: Source language code (default: `en-us`)
- `-t, --target`: Target language code (default: `cmn` for Mandarin Chinese)
- `-m, --model`: Ollama model to use (default: `qwen3:30b-a3b-instruct-2507-q4_K_M`)
- `--threshold`: Phonetic similarity threshold 0-1 (default: `0.5`)
- `--save-dir`: Directory to save results (default: `outputs`)

**Requirements:**
- [Ollama](https://ollama.com/) installed and running
- espeak-ng installed for IPA analysis
- Recommended model: `ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M`

## Acknowledgments

- [Demucs](https://github.com/facebookresearch/demucs) by Facebook Research
- [XTTS](https://github.com/coqui-ai/TTS) by Coqui AI

## License

Apache License 2.0
