# BLT Benchmarks

Benchmarking tools for translation quality evaluation.

## Quick Start

### 1. Run Experiments

```bash
# Single language pair
python -m benchmarks.run_experiment cmn en-us

# All pairs
python -m benchmarks.run_experiment --all
```

### 2. Generate Reports & Visualizations

```bash
python -m benchmarks.generate_reports
python -m benchmarks.generate_visualizations
```

### 3. View Results

```bash
cat benchmarks/results/cmn→en-us_agent.md
ls assets/visualization/
```

## Metrics

- **SER** - Syllable Error Rate (lower is better)
- **SCRE** - Syllable Count Relative Error (lower is better)
- **ARI** - Adjusted Rand Index for rhyme clustering (higher is better)

See [experiments/README.md](experiments/README.md) for details.

## Setup

```bash
# Ollama
ollama serve
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M

# Python
cd blt
uv sync --dev
```

## File Structure

```
benchmarks/
├── data/                      # Scraped lyrics
├── utils/                     # Scraper tools
├── experiments/               # Evaluation framework
├── results/                   # Output reports
├── run_experiment.py          # Run experiments
├── generate_reports.py        # Generate comparison reports
└── generate_visualizations.py # Generate charts
```

## License

Same as parent BLT project.
