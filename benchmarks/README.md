# BLT Benchmarks

Comprehensive benchmarking tools for the BLT translation agent, including data collection and quality evaluation.

## Overview

This benchmarking suite provides:

1. **Data Collection** - Scrape multilingual lyrics from KKBOX
2. **Quality Experiments** - Compare agent vs baseline translation quality
3. **Research Reports** - Detailed metrics and analysis

## Quick Start

### 1. Collect Data

Scrape multilingual lyrics datasets:

```bash
# Chinese lyrics
python -m benchmarks.utils.kkbox_scraper \
  https://www.kkbox.com/tw/tc/artist/StHnkq8RxUXeTrn7ij/albums \
  --output-file cmn_lyrics.json

# English lyrics
python -m benchmarks.utils.kkbox_scraper \
  https://www.kkbox.com/tw/tc/artist/4p8HBNPfLgp4cPr3Lk/albums \
  --output-file en_lyrics.json

# Japanese lyrics
python -m benchmarks.utils.kkbox_scraper \
  https://www.kkbox.com/tw/tc/artist/-nvBpZSsLGYcEJknDO/albums \
  --output-file ja_lyrics.json
```

### 2. Run Experiments

Test translation quality:

```bash
# Single language pair (Chinese → English)
python -m benchmarks.run_experiment cmn en-us --samples 10

# All language pairs
python -m benchmarks.run_experiment --all --samples 10
```

### 3. Review Results

Reports saved to `benchmarks/results/`:
- `*.json` - Raw metrics data
- `*.md` - Human-readable analysis

---

## Components

### 🎵 KKBOX Lyrics Scraper

Collect multilingual lyrics datasets for benchmarking.

#### Features

- **Album Page Support**: Pass `/albums` URLs to scrape all songs from all albums
- **Automatic Lyrics Cleaning**: Removes metadata (作詞, 作曲, etc.)
- **Language Detection**: Auto-detects Chinese, Japanese, or English
- **Async Scraping**: Fast concurrent requests with rate limiting
- **Deduplication**: Appends only new songs to existing files

#### Usage

```bash
python -m benchmarks.utils.kkbox_scraper <URL> [OPTIONS]

Options:
  --output DIR            Output directory (default: benchmarks/data)
  --output-file FILE      Output filename (default: cmn_lyrics.json)
  --max-concurrent N      Max concurrent requests (default: 3)
  --rate-limit-delay N    Delay between requests in seconds (default: 1.0)
```

---

### 🔬 Translation Quality Experiments

Compare constraint-aware agent vs baseline neural MT.

#### Evaluated Metrics

1. **Syllable Count Accuracy** - % of lines with exact syllable match
2. **Syllable Pattern Similarity** - Rhythm preservation (0-1 score)
3. **Rhyme Preservation** - % of rhyme pairs maintained

#### Usage

```bash
# Run experiment for Chinese → English
python -m benchmarks.run_experiment cmn en-us --samples 10

# Run all language pairs (cmn↔en, cmn↔ja, en↔ja)
python -m benchmarks.run_experiment --all --samples 10

# Custom model
python -m benchmarks.run_experiment ja en-us --model llama3:70b --samples 5
```

#### Language Pairs Supported

- `cmn` (Chinese) ↔ `en-us` (English)
- `cmn` ↔ `ja` (Japanese)
- `en-us` ↔ `ja`

**Total**: 6 directional pairs (e.g., cmn→en and en→cmn are separate experiments)

📖 **[Full Experiments Documentation](experiments/README.md)**

---

## Project Structure

```
benchmarks/
├── README.md                     # This file
├── run_experiment.py             # Experiment CLI
│
├── data/                         # Scraped lyrics datasets
│   ├── cmn_lyrics.json           # Chinese songs (91 songs)
│   ├── en_lyrics.json            # English songs (82 songs)
│   └── ja_lyrics.json            # Japanese songs (102 songs)
│
├── utils/                        # Data collection tools
│   ├── __init__.py
│   └── kkbox_scraper.py          # KKBOX scraper
│
├── experiments/                  # Quality evaluation framework
│   ├── __init__.py
│   ├── evaluator.py              # Metrics calculation
│   ├── baseline.py               # Baseline translator (no agent)
│   ├── runner.py                 # Experiment orchestration
│   ├── reporter.py               # Report generation
│   ├── utils.py                  # Test case sampling
│   └── README.md                 # Experiments docs
│
└── results/                      # Experiment outputs
    ├── *.json                    # Raw results
    └── *.md                      # Reports
```

---

## Research Workflow

### End-to-End Example

```bash
# 1. Collect data (if not already done)
python -m benchmarks.utils.kkbox_scraper \
  https://www.kkbox.com/tw/tc/artist/StHnkq8RxUXeTrn7ij/albums \
  --output-file cmn_lyrics.json

# 2. Run experiment
python -m benchmarks.run_experiment cmn en-us --samples 20

# 3. Review results
cat benchmarks/results/cmn→en-us_20samples.md
```

### Typical Results

Based on pilot testing with Qwen 30B model:

| Metric | Agent | Baseline | Improvement |
|--------|-------|----------|-------------|
| Syllable Accuracy | ~90% | ~60% | +30% |
| Pattern Similarity | ~0.85 | ~0.50 | +0.35 |
| Rhyme Preservation | ~80% | ~45% | +35% |
| **Success Rate** | **~85%** | **~30%** | **+55%** |

**Trade-off**: Agent takes ~5x longer but produces significantly higher quality constraint-preserving translations.

---

## Prerequisites

### Ollama Setup

```bash
# Ensure Ollama is running
ollama serve

# Pull required model
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
```

### Python Dependencies

```bash
cd /home/gmchiu/Documents/GitHub/blt
uv sync
```

---

## Advanced Usage

### Programmatic API

```python
from benchmarks.experiments import (
    ExperimentRunner,
    ComparisonReporter,
    sample_test_cases,
    load_lyrics_from_json,
)

# Load data
lyrics = load_lyrics_from_json("benchmarks/data/cmn_lyrics.json", "cmn")

# Create test cases
test_cases = sample_test_cases(
    source_lyrics=lyrics,
    source_lang="cmn",
    target_lang="en-us",
    num_samples=10,
)

# Run experiment
runner = ExperimentRunner()
results = runner.run_experiment(test_cases)

# Generate report
reporter = ComparisonReporter()
reporter.generate_markdown_report(results, "my_report.md")
```

See [experiments/README.md](experiments/README.md) for full API documentation.

---

## Troubleshooting

### Ollama Connection Error

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Model Not Found

```bash
# Pull the model
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
```

### Import Errors

```bash
# Ensure you're in project root
cd /home/gmchiu/Documents/GitHub/blt

# Sync dependencies
uv sync

# Run from project root
python -m benchmarks.run_experiment ...
```

---

## License

Same as parent BLT project.
