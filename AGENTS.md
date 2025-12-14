# Development Guide

## Setup

```bash
# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate
uv sync

# Configure espeak-ng (required for IPA phonetic analysis)
export PHONEMIZER_ESPEAK_PATH="/opt/homebrew/bin/espeak-ng"
export PHONEMIZER_ESPEAK_LIBRARY="/opt/homebrew/lib/libespeak-ng.dylib"
```

## Testing

```bash
uv run python -m pytest                      # Run all tests
uv run python -m pytest tests/test_file.py   # Run specific test file
```

## Development Rules

**Dependencies:**
- Use `uv add <package>` to add dependencies (not pip)
- Use `uv sync` to sync dependencies

**Code Quality:**
- Use `ruff` for linting and formatting
- Follow type hints (Python 3.11+)

**Performance:**
- Lazy-load heavy models (HanLP, phonemizer)
- Cache expensive computations where appropriate
