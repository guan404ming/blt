# OMG: Optimized Music Generation

[![Python 3.11](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**OMG** (Optimized Music Generation) is a modular pipeline for enhancing Text-to-Music generation. It integrates different optimization blocks that can be combined to improve MusicGen output quality.

### Pipeline Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│   Prompt    │ ──► │   Retrieval  │ ──► │   MusicGen  │ ──► │  Output  │
│             │     │    (CLAP)    │     │    (ICL)    │     │          │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────┘
```

### Optimization Blocks

| Block                 | Description                                            | Status  |
| --------------------- | ------------------------------------------------------ | ------- |
| CLAP Retrieval        | Retrieve similar examples via embedding similarity     | Done    |
| In-Context Learning   | Condition generation on retrieved examples             | Done    |
| RAG                   | Retrieval-augmented generation with external knowledge | Planned |
| LoRA Fine-tuning      | Parameter-efficient adaptation for specific styles     | Planned |
| Prompt Enhancement    | LLM-based prompt refinement                            | Planned |
| Audio Post-processing | Style transfer, mixing                                 | Planned |
| Lyrics Translation    | Cover song generation with new lyrics                  | Done    |
| Evaluation            | CLAP score                                             | Done    |

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

## Data Preparation

```bash
# Download MusicCaps dataset and audio
uv run python scripts/load_data.py

# Create caption mappings and embeddings
uv run python scripts/create_audio_caption_mapping.py
```

## Usage

```bash
uv run python scripts/poc.py \
    --prompt "jazz with only saxophone and keyboard interplay" \
    --duration 20 \
    --top-k 3 \
    --threshold 0.7
```

**Parameters:**

- `--prompt`: Text description for music generation
- `--duration`: Audio length in seconds (default: 20)
- `--top-k`: Number of examples to retrieve (default: 3)
- `--threshold`: Minimum similarity score (default: 0.7)
- `--no-icl`: Skip in-context learning

## Citation

```bibtex
@misc{omg2025,
    title={OMG: Optimized Music Generation with In-Context Learning},
    author={Guan-Ming Chiu},
    year={2025},
    url={https://github.com/guan404ming/omg}
}
```

## Acknowledgments

- [CLAP](https://github.com/LAION-AI/CLAP) by LAION AI

## License

Apache License 2.0
