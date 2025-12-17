"""
KKBOX Lyrics Scraper

Modern async web scraper for collecting lyrics data from KKBOX.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup


class KKBOXScraper:
    """Async scraper for KKBOX artist pages and lyrics"""

    def __init__(
        self,
        output_dir: str = "benchmarks/data",
        output_file: str = "cmn_lyrics.json",
        max_concurrent: int = 3,
        rate_limit_delay: float = 1.0,
        skip_existing: bool = True,
    ):
        """
        Initialize scraper

        Args:
            output_dir: Directory to save scraped data
            output_file: Output filename (default: cmn_lyrics.json)
            max_concurrent: Maximum concurrent requests
            rate_limit_delay: Delay between requests (seconds)
            skip_existing: Skip songs that already exist in output file (default: True)
        """
        self.output_dir = Path(output_dir)
        self.output_file = output_file
        self.output_path = self.output_dir / output_file
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        self.skip_existing = skip_existing
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Load existing data
        self.existing_songs = self._load_existing_songs()

        # HTTP client with realistic headers
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        }

    async def scrape_artist(self, artist_url: str) -> dict:
        """
        Scrape all lyrics from an artist page or albums page

        Args:
            artist_url: KKBOX artist page URL (can be /artist/ or /artist/.../albums)

        Returns:
            Dictionary with artist info and scraped songs
        """
        print(f"🎵 Scraping: {artist_url}")

        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            # Get page
            response = await client.get(artist_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Extract artist name
            artist_name = self._extract_artist_name(soup)
            print(f"   Artist: {artist_name}")

            # Check if this is an albums page
            if "/albums" in artist_url:
                song_urls = await self._extract_songs_from_albums(
                    client, soup, artist_url
                )
            else:
                # Regular artist page - extract song links directly
                song_urls = self._extract_song_urls(soup, artist_url)

            print(f"   Found {len(song_urls)} songs")

            # Scrape each song with rate limiting
            songs = []
            skipped_count = 0
            for i, song_url in enumerate(song_urls, 1):
                # Check if already exists in output file
                if self._song_url_exists(song_url) and self.skip_existing:
                    print(
                        f"   [{i}/{len(song_urls)}] ⏭️  Skipped (already in file): {song_url}"
                    )
                    skipped_count += 1
                    continue

                print(f"   [{i}/{len(song_urls)}] Scraping: {song_url}")

                song_data = await self._scrape_song_with_limit(client, song_url)

                if song_data:
                    songs.append(song_data)
                    # Add to existing songs list
                    self.existing_songs.append(song_data)

                # Rate limiting
                if i < len(song_urls):
                    await asyncio.sleep(self.rate_limit_delay)

            if skipped_count > 0:
                print(
                    f"✅ Successfully scraped {len(songs)}/{len(song_urls)} new songs ({skipped_count} skipped)"
                )
            else:
                print(
                    f"✅ Successfully scraped {len(songs)}/{len(song_urls)} new songs"
                )

            # Save all songs to file
            self.save_all_songs()

            return {
                "artist_name": artist_name,
                "new_songs": len(songs),
                "total_songs": len(self.existing_songs),
            }

    async def _scrape_song_with_limit(
        self, client: httpx.AsyncClient, song_url: str
    ) -> Optional[dict]:
        """Scrape a single song with semaphore-based rate limiting"""
        async with self.semaphore:
            return await self._scrape_song(client, song_url)

    async def _scrape_song(
        self, client: httpx.AsyncClient, song_url: str
    ) -> Optional[dict]:
        """
        Scrape lyrics from a single song page

        Args:
            client: HTTP client
            song_url: Song page URL

        Returns:
            Dictionary with song data or None if failed
        """
        try:
            response = await client.get(song_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Extract song title
            song_title = self._extract_song_title(soup)

            # Extract artist name from song page
            artist_name = self._extract_artist_from_song_page(soup)

            # Extract lyrics (try JSON-LD first, fallback to HTML)
            lyrics = self._extract_lyrics(soup)

            if not lyrics:
                print(f"      ⚠️  No lyrics found for: {song_title}")
                return None

            # Detect language
            language = self._detect_language(lyrics)

            song_data = {
                "song_title": song_title,
                "artist_name": artist_name,
                "lyrics": lyrics,
                "language": language,
                "url": song_url,
                "line_count": len(
                    [line for line in lyrics.split("\n") if line.strip()]
                ),
            }

            return song_data

        except Exception as e:
            print(f"      ❌ Error scraping {song_url}: {e}")
            return None

    async def _extract_songs_from_albums(
        self, client: httpx.AsyncClient, soup: BeautifulSoup, base_url: str
    ) -> list[str]:
        """
        Extract all song URLs from albums page

        First extracts all album URLs, then visits each album to get song URLs
        """
        # Extract album URLs
        album_urls = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/album/" in href and href not in album_urls:
                full_url = urljoin(base_url, href)
                album_urls.append(full_url)

        print(f"   Found {len(album_urls)} albums")

        # Extract songs from each album
        all_song_urls = []
        for i, album_url in enumerate(album_urls, 1):
            print(f"   [{i}/{len(album_urls)}] Scanning album: {album_url}")

            try:
                response = await client.get(album_url)
                response.raise_for_status()
                album_soup = BeautifulSoup(response.text, "lxml")

                # Extract song URLs from this album
                song_urls = self._extract_song_urls(album_soup, album_url)
                all_song_urls.extend(song_urls)

                # Rate limiting between album fetches
                if i < len(album_urls):
                    await asyncio.sleep(self.rate_limit_delay)

            except Exception as e:
                print(f"      ⚠️  Error scanning album: {e}")

        # Deduplicate
        return list(dict.fromkeys(all_song_urls))

    def _extract_artist_name(self, soup: BeautifulSoup) -> str:
        """Extract artist name from artist page"""
        # Try h1 first
        h1 = soup.find("h1")
        if h1:
            # Remove follower count and clean up
            text = h1.get_text(strip=True)
            # Remove patterns like "2.52萬位粉絲"
            text = re.sub(r"\d+[\.,\d]*[萬千百]*位粉絲", "", text)
            return text.strip()

        return "Unknown Artist"

    def _extract_song_urls(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract all song URLs from artist page"""
        song_urls = []

        # Find all links with /song/ pattern
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/song/" in href:
                # Convert to absolute URL
                full_url = urljoin(base_url, href)
                # Deduplicate
                if full_url not in song_urls:
                    song_urls.append(full_url)

        return song_urls

    def _extract_song_title(self, soup: BeautifulSoup) -> str:
        """Extract song title from song page"""
        # Try h1 first (song title is in h1 on KKBOX)
        h1 = soup.find("h1")
        if h1:
            text = h1.get_text(strip=True)
            # Skip if it's the artist name or follower count
            if "位粉絲" not in text and len(text) < 100:
                return text

        # Fallback to title tag
        title = soup.find("title")
        if title:
            # Extract from format like "山海 - 草東沒有派對 (No Party For Cao Dong) - KKBOX"
            text = title.get_text(strip=True)
            if " - " in text:
                return text.split(" - ")[0].strip()

        return "Unknown Title"

    def _extract_artist_from_song_page(self, soup: BeautifulSoup) -> str:
        """Extract artist name from song page"""
        # Look for artist link
        for link in soup.find_all("a", href=True):
            if "/artist/" in link["href"]:
                return link.get_text(strip=True)

        return "Unknown Artist"

    def _extract_lyrics(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract lyrics from song page

        Tries JSON-LD schema first, then falls back to HTML parsing
        """
        lyrics_text = None

        # Try JSON-LD first
        json_ld = soup.find("script", {"type": "application/ld+json"})
        if json_ld:
            try:
                data = json.loads(json_ld.string)
                if isinstance(data, dict):
                    # Navigate to lyrics.text
                    if "lyrics" in data and isinstance(data["lyrics"], dict):
                        lyrics_text = data["lyrics"].get("text")
            except json.JSONDecodeError:
                pass

        # Fallback: look for common lyrics containers
        if not lyrics_text:
            for pattern in ["lyric", "lyrics", "歌詞"]:
                div = soup.find("div", {"class": re.compile(pattern, re.I)})
                if div:
                    text = div.get_text(separator="\n", strip=True)
                    if len(text) > 50:  # Sanity check
                        lyrics_text = text
                        break

        # Last resort: look for pre tags (sometimes used for lyrics)
        if not lyrics_text:
            pre = soup.find("pre")
            if pre:
                lyrics_text = pre.get_text(strip=True)

        # Clean up lyrics (remove credits)
        if lyrics_text:
            return self._clean_lyrics(lyrics_text)

        return None

    def _clean_lyrics(self, lyrics: str) -> str:
        """
        Clean up lyrics by removing credits and metadata

        Removes:
        - 作詞：... (Lyricist)
        - 作曲：... (Composer)
        - 編曲：... (Arranger)
        - 製作：... (Producer)
        """
        lines = lyrics.split("\n")
        cleaned_lines = []

        for line in lines:
            # Skip credit lines (作詞, 作曲, 編曲, 製作, etc.)
            if re.match(r"^\s*(作詞|作曲|編曲|製作|監製|主唱|演唱)[:：]", line):
                continue
            # Skip empty lines at the start
            if not cleaned_lines and not line.strip():
                continue
            cleaned_lines.append(line)

        # Join and clean up extra whitespace
        result = "\n".join(cleaned_lines).strip()
        # Remove multiple consecutive blank lines
        result = re.sub(r"\n\s*\n\s*\n+", "\n\n", result)

        return result

    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on character composition

        Returns:
            Language code (cmn, en-us, ja, etc.)
        """
        # Count character types
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        japanese_chars = len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff]", text))
        english_chars = len(re.findall(r"[a-zA-Z]", text))

        total = chinese_chars + japanese_chars + english_chars
        if total == 0:
            return "unknown"

        # Determine primary language
        if chinese_chars / total > 0.5:
            return "cmn"
        elif japanese_chars / total > 0.3:
            return "ja"
        elif english_chars / total > 0.5:
            return "en-us"
        else:
            return "mixed"

    def _load_existing_songs(self) -> list[dict]:
        """
        Load existing songs from output file

        Returns:
            List of existing song dictionaries
        """
        if self.output_path.exists():
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(
                            f"📂 Loaded {len(data)} existing songs from {self.output_path}"
                        )
                        return data
            except json.JSONDecodeError:
                print(f"⚠️  Could not parse {self.output_path}, starting fresh")
        return []

    def _song_url_exists(self, url: str) -> bool:
        """
        Check if a song URL already exists in existing songs

        Args:
            url: Song URL to check

        Returns:
            True if URL exists, False otherwise
        """
        return any(song.get("url") == url for song in self.existing_songs)

    def save_all_songs(self) -> None:
        """
        Save all songs to the output file

        Format: benchmarks/data/cmn_lyrics.json as array
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save as array
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.existing_songs, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved {len(self.existing_songs)} total songs to {self.output_path}")


def parse_args():
    """Parse command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape lyrics data from KKBOX artist pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape from an artist page
  python -m benchmarks.utils.kkbox_scraper https://www.kkbox.com/tw/tc/artist/StHnkq8RxUXeTrn7ij

  # Scrape from albums page (recommended - gets more songs)
  python -m benchmarks.utils.kkbox_scraper https://www.kkbox.com/tw/tc/artist/StHnkq8RxUXeTrn7ij/albums

  # Custom output directory and concurrency
  python -m benchmarks.utils.kkbox_scraper https://www.kkbox.com/tw/tc/artist/StHnkq8RxUXeTrn7ij/albums \\
      --output benchmarks/data \\
      --concurrent 5 \\
      --delay 0.5
        """,
    )

    parser.add_argument(
        "artist_url",
        help="KKBOX artist or albums page URL (e.g., https://www.kkbox.com/tw/tc/artist/... or .../albums)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="benchmarks/data",
        help="Output directory for scraped data (default: benchmarks/data)",
    )

    parser.add_argument(
        "--output-file",
        default="cmn_lyrics.json",
        help="Output filename (default: cmn_lyrics.json)",
    )

    parser.add_argument(
        "--concurrent",
        "-c",
        type=int,
        default=3,
        help="Maximum concurrent requests (default: 3)",
    )

    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-scraping even if files already exist",
    )

    return parser.parse_args()


async def main():
    """Main entry point"""
    import sys

    args = parse_args()

    # Validate URL
    if "/artist/" not in args.artist_url:
        print("❌ Error: URL must be a KKBOX artist or albums page (/artist/ in URL)")
        sys.exit(1)

    print("🎵 KKBOX Lyrics Scraper")
    print(f"   Artist URL: {args.artist_url}")
    print(f"   Output: {args.output}")
    print(f"   Output file: {args.output_file}")
    print(f"   Max concurrent: {args.concurrent}")
    print(f"   Rate limit delay: {args.delay}s")
    print(f"   Skip existing: {not args.force}")
    print()

    # Create scraper
    scraper = KKBOXScraper(
        output_dir=args.output,
        output_file=args.output_file,
        max_concurrent=args.concurrent,
        rate_limit_delay=args.delay,
        skip_existing=not args.force,
    )

    # Scrape artist
    try:
        result = await scraper.scrape_artist(args.artist_url)

        print("\n✅ Scraping complete!")
        print(f"   Artist: {result['artist_name']}")
        print(f"   New songs: {result['new_songs']}")
        print(f"   Total songs in file: {result['total_songs']}")
        print(f"   Data saved to: {scraper.output_path}")

    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
