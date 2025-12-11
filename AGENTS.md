# AGENTS.md - BLT (Better Lyrics Translation Toolkit)

## Project Overview

Python toolkit for song translation with music constraint preservation using **pydantic-ai** agents and **IPA phonemic analysis**.

**Core principle**: Agent-based translation loop with verification tools, not direct LLM translation.

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

**Required environment variables:**
```bash
export GOOGLE_API_KEY="your-api-key"
export PHONEMIZER_ESPEAK_PATH="/opt/homebrew/bin/espeak-ng"
export PHONEMIZER_ESPEAK_LIBRARY="/opt/homebrew/lib/libespeak-ng.dylib"
```

## Testing

```bash
uv run python -m pytest                        # All tests
uv run python -m pytest tests/test_translator.py  # Specific file
```

## Critical Context

### Constraint Priority
1. **Syllable patterns** (word-level rhythm) - HIGHEST
2. **Total syllable count** - CRITICAL
3. **Rhyme scheme** - OPTIONAL

**Why**: Singing requires matching rhythmic feel, not just total duration.

### Agent Pattern
```
1. Draft translation
2. Call verify_all_constraints() tool
3. Read feedback: "Line 1: expected [2,2,1] but got [1,3,1]"
4. Restructure to match pattern
5. Repeat (max 15x)
```

**Never verify constraints via LLM reasoning - always use tools.**

### Tool Registration Order
```python
config.register_tools(agent, analyzer, validator)  # Register first
agent._system_prompt = config.get_system_prompt()  # Then set prompt
```

## Key Rules

- ✅ Use `uv add` for dependencies (not pip)
- ✅ Lazy-load heavy models (HanLP, phonemizer)
- ✅ Prioritize syllable patterns over total count
- ❌ Don't modify `_system_prompt` before `register_tools()` completes
- ❌ Don't validate constraints without tools
