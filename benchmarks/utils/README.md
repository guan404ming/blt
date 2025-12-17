# BLT Benchmarking System

This directory contains the benchmarking infrastructure for the BLT (Better Lyrics Translation) toolkit.

## Structure

```
benchmarks/
├── data/                    # Scraped lyrics data
│   ├── [song_title].json   # Individual song files
│   └── _summary.json       # Artist summary
├── utils/                   # Utilities
│   ├── kkbox_scraper.py    # KKBOX web scraper
│   └── __init__.py
└── README.md               # This file
```

## KKBOX Lyrics Scraper

### Features

- **Album Page Support**: Scrape all songs from all albums via `/albums` page
- **Artist Page Support**: Scrape songs directly from artist page
- **Automatic Lyrics Cleaning**: Removes credits (作詞/作曲/編曲/製作)
- **Language Detection**: Auto-detects Chinese, Japanese, English, or mixed
- **Async Scraping**: Fast concurrent requests with rate limiting
- **Deduplication**: Automatically removes duplicate songs

### Quick Start

Scrape songs from an artist page:

```bash
python -m benchmarks.utils.kkbox_scraper https://www.kkbox.com/tw/tc/artist/StHnkq8RxUXeTrn7ij
```

Scrape all songs from all albums (recommended - gets more songs):

```bash
python -m benchmarks.utils.kkbox_scraper https://www.kkbox.com/tw/tc/artist/StHnkq8RxUXeTrn7ij/albums
```

### Options

```bash
python -m benchmarks.utils.kkbox_scraper --help
```

- `--output, -o`: Output directory (default: `benchmarks/data`)
- `--concurrent, -c`: Max concurrent requests (default: 3)
- `--delay, -d`: Delay between requests in seconds (default: 1.0)

### Examples

```bash
# Basic usage
python -m benchmarks.utils.kkbox_scraper https://www.kkbox.com/tw/tc/artist/StHnkq8RxUXeTrn7ij

# Custom settings
python -m benchmarks.utils.kkbox_scraper https://www.kkbox.com/tw/tc/artist/StHnkq8RxUXeTrn7ij/albums \
    --output benchmarks/data \
    --concurrent 5 \
    --delay 0.5
```

## Output Format

Each song is saved as a JSON file:

```json
{
  "song_title": "山海",
  "artist_name": "草東沒有派對",
  "lyrics": "...",
  "language": "cmn",
  "url": "https://www.kkbox.com/tw/tc/song/...",
  "line_count": 22
}
```

An artist summary is also saved:

```json
{
  "artist_name": "草東沒有派對 (No Party For Cao Dong)",
  "artist_url": "https://www.kkbox.com/tw/tc/artist/...",
  "total_songs": 5,
  "songs": [...]
}
```

## Language Detection

The scraper automatically detects the language based on character composition:
- `cmn`: Chinese (Mandarin)
- `ja`: Japanese
- `en-us`: English
- `mixed`: Mixed languages

## Rate Limiting

The scraper includes built-in rate limiting to be respectful to KKBOX servers:
- Default: 3 concurrent requests
- Default: 1 second delay between requests
- Configurable via CLI options

## Dependencies

The scraper requires the following dev dependencies (already added to `pyproject.toml`):
- `httpx` - Async HTTP client
- `beautifulsoup4` - HTML parsing
- `lxml` - Fast HTML/XML parser

Install with:
```bash
uv sync --dev
```

## Future Features

- [ ] Benchmark runner
- [ ] Evaluation metrics
- [ ] Report generation
- [ ] Baseline comparison
- [ ] Visualization
