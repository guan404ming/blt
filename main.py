#!/usr/bin/env python3
"""Main entry point for Audio ICL project."""


def main():
    """Main entry point."""
    print("Audio In-Context Learning for Text-to-Music Generation")
    print("=" * 55)
    print()
    print("Available scripts:")
    print("  - scripts/poc_inference.py  : Run PoC inference")
    print()
    print("Usage:")
    print("  uv run python scripts/poc_inference.py")
    print()
    print("For development:")
    print("  uv sync --dev")
    print("  uv run pytest tests/")


if __name__ == "__main__":
    main()
