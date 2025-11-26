# Project Roadmap & Architecture

This document details the architecture of the BLT project and the roadmap for its "Song Translation" feature.

## Pipeline Architecture

The core of BLT is a pipeline that transforms an original song into a translated version while preserving the original vocal style.

```
┌──────────────────┐     ┌─────────────────┐     ┌───────────────────┐     ┌───────────┐
│  Original Song   │ ──► │ Vocal Separator │ ──► │  Lyrics Aligner   │ ──► │   Voice   │
│      Audio       │     │    (Demucs)     │     │      (Word-Level) │     │ Synthesizer│
└──────────────────┘     └─────────────────┘     └───────────────────┘     └───────────┘
        │                                                  ▲                   ▲
        │                                                  │                   │
        ▼                                          ┌───────────────┐           │
┌──────────────────┐     ┌─────────────────┐       │  Translated   │           │
│ Instrumental Mix │ ◄── │   Audio Mixer   │ ◄──── │    Lyrics     │ ──────────┘
└──────────────────┘     └─────────────────┘       │     (LLM)     │
                                                   └───────────────┘
```

### Pipeline Steps
1.  **Vocal Separation**: We use **Demucs** to split the original song into instrumental and vocal tracks.
2.  **Lyrics Alignment**: The system aligns the original lyrics with the isolated vocal track to understand the precise timing of each word.
3.  **Lyrics Translation**: An LLM-based engine translates the lyrics, strictly adhering to musical constraints (syllables, rhyme) to ensure the new lyrics fit the melody.
4.  **Voice Synthesis**: A voice cloning model (**XTTS**) generates the singing voice for the new lyrics, mimicking the timbre of the original artist.
5.  **Audio Mixing**: The new vocals are mixed back with the original instrumental track to create the final cover song.

---

## Current Implementation: BLT Engine

The **BLT** engine is designed to produce "singable" lyrics directly, without requiring a separate melody extraction step.

### Core Features
-   **LLM-Driven**: Uses models like **Gemini Pro** or **GPT-4** for high-quality translation.
-   **Constraint Satisfaction**:
    -   **Syllable Count**: Matches the target language syllables to the original line's count.
    -   **Rhyme Scheme**: Detects and attempts to preserve the original rhyme pattern (e.g., AABB).
-   **Feedback Loop**: If the LLM generates lyrics that violate constraints (e.g., too many syllables), the system validates the output and automatically re-prompts the model to fix the errors.

### Tech Stack
-   **PydanticAI**: Ensures structured output from the LLM for easier validation.
-   **Feature Extractor**: Automatically counts syllables and detects rhyme schemes from text.

---

## Roadmap

### Short-term Goals
-   [ ] **Improve Rhyme Accuracy**: Enhance the rhyme detection and generation logic for non-English languages.
-   [ ] **Word Boundary Control**: Better alignment of translated words with musical pauses.
-   [ ] **Validation Metrics**: Implement automated scoring for "singability" (e.g., rhythm alignment).

### Long-term Goals
-   **Melody-Aware Translation**: Feed audio features (pitch, rhythm) directly into the translation model.
-   **Real-time Translation**: optimize the pipeline for near real-time performance.
-   **Multi-Singer Support**: Better handling of duets or songs with backing vocals.