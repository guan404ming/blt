# Build & Publish

## Automated Release (Recommended)

**Setup PyPI Trusted Publishing:** https://pypi.org/manage/account/publishing/
```
Project: blt-toolkit | Owner: guan404ming | Repo: blt
Workflow: publish.yml | Environment: pypi
```

**Create GitHub Environment:** https://github.com/guan404ming/blt/settings/environments

**Release:**
```bash
# Update version in pyproject.toml, then:
git commit -am "Release vX.Y.Z" && git tag vX.Y.Z && git push origin main --tags
```

## Manual Publish

```bash
rm -rf dist/ && uv run python -m build && uv run twine upload dist/*
```

Get PyPI token: https://pypi.org/manage/account/token/

## Test Install

```bash
pip install blt-toolkit
python -c "from blt.translators import LyricsTranslator; print('OK')"
```
