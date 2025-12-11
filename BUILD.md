# Build & Publish Guide

## Build Package

```bash
# Clean and build
rm -rf dist/ build/
uv run python -m build

# Verify
uv run twine check dist/*
```

## Publish via GitHub Actions (Recommended)

### 1. Setup PyPI Trusted Publishing

Go to https://pypi.org/manage/account/publishing/ and add:

```
PyPI Project Name: blt-toolkit
Owner: guan404ming
Repository: blt
Workflow name: publish.yml
Environment name: pypi
```

### 2. Create GitHub Environment

Go to https://github.com/guan404ming/blt/settings/environments

- Create environment: `pypi`

### 3. Release

```bash
# Update version in pyproject.toml
git add pyproject.toml
git commit -m "Release v0.1.0"
git tag v0.1.0
git push origin main --tags
```

GitHub Actions automatically builds and publishes!

## Manual Publish (Backup Method)

```bash
# Build
rm -rf dist/
uv run python -m build

# Upload (requires PyPI token)
uv run twine upload dist/*
```

Get PyPI token at: https://pypi.org/manage/account/token/

## Quick Reference

**Automated:**
```bash
# Update version in pyproject.toml
git tag v0.2.0
git push origin --tags
```

**Manual:**
```bash
rm -rf dist/
uv run python -m build
uv run twine upload dist/*
```

**Test Install:**
```bash
pip install blt-toolkit
python -c "from blt.translators import LyricsTranslator; print('OK')"
```
