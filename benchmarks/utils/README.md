# KKBOX Lyrics Scraper

Scrape multilingual lyrics datasets from KKBOX.

## Usage

```bash
# Album page (gets all songs from all albums)
python -m benchmarks.utils.kkbox_scraper https://www.kkbox.com/tw/tc/artist/StHnkq8RxUXeTrn7ij/albums

# Artist page (gets songs directly)
python -m benchmarks.utils.kkbox_scraper https://www.kkbox.com/tw/tc/artist/StHnkq8RxUXeTrn7ij

# Custom options
python -m benchmarks.utils.kkbox_scraper <URL> \
  --output benchmarks/data \
  --concurrent 5 \
  --delay 0.5
```

## Output

Songs saved as JSON with metadata:
- `song_title` - Track name
- `artist_name` - Artist name
- `lyrics` - Cleaned lyrics (credits removed)
- `language` - Auto-detected (cmn, ja, en-us, mixed)
- `url` - KKBOX link
- `line_count` - Number of lines

## Features

- Album/artist page support
- Automatic lyrics cleaning
- Language auto-detection
- Async scraping with rate limiting
- Deduplication

