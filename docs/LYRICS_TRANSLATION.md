# Lyrics Translation & Cover Song Generation

This module provides functionality to generate cover songs with new lyrics while maintaining the original singer's voice characteristics and instrumental.

## Overview

**Input:**
- Original song audio (with vocals and instrumental)
- Original lyrics
- New lyrics

**Output:**
- Cover song with new lyrics sung in the original singer's voice
- Separated vocals and instrumental tracks
- Lyrics-to-audio alignment data

## Architecture

The pipeline consists of four main components:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Original Song  │────▶│ Vocal Separator  │────▶│    Vocals +     │
│     Audio       │     │    (Demucs)      │     │  Instrumental   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
┌─────────────────┐                                       │
│  Old Lyrics     │──────────────────────────────────────┐│
└─────────────────┘                                       ││
                                                          ▼▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │ Lyrics Aligner   │────▶│  Word Timings   │
                        │  (Forced Align)  │     │                 │
                        └──────────────────┘     └────────┬────────┘
                                                           │
┌─────────────────┐                                       │
│  New Lyrics     │──────────────────────────────────────┐│
└─────────────────┘                                       ││
                                                          ▼▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │ Voice Synthesizer│────▶│   New Vocals    │
                        │ (TTS + VC)       │     │                 │
                        └──────────────────┘     └────────┬────────┘
                                                           │
                        ┌──────────────────┐              │
                        │  Instrumental    │──────────────┤
                        └──────────────────┘              │
                                                          ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │     Mixer        │────▶│  Final Cover    │
                        │                  │     │     Song        │
                        └──────────────────┘     └─────────────────┘
```

### 1. Vocal Separator
- **Technology:** Meta's Demucs (Deep Extractor for Music Sources)
- **Function:** Separates audio into vocals and instrumental tracks
- **Model:** `htdemucs` (default), `htdemucs_ft`, or `mdx_extra`

### 2. Lyrics Aligner
- **Technology:** CTC-based forced alignment with wav2vec2/MMS
- **Function:** Aligns lyrics words with their timing in the audio
- **Model:** `MahmoudAshraf/mms-300m-1130-forced-aligner` (default)
- **Output:** Word-level timing information

### 3. Voice Synthesizer
- **Technology:** TTS + Voice Conversion
- **Function:** Synthesizes new vocals with the target lyrics
- **Features:**
  - Voice cloning from reference audio
  - Pitch manipulation
  - Time stretching
  - Prosody transfer

### 4. Mixer
- **Function:** Combines synthesized vocals with original instrumental
- **Features:**
  - Volume adjustment
  - Normalization
  - Channel matching

## Installation

The required dependencies are already included in the project:

```bash
# Already installed
uv add demucs ctc-forced-aligner praat-parselmouth
```

For production-grade singing voice synthesis, consider installing additional models:

```bash
# Optional: seed-vc for singing voice conversion
pip install seed-vc

# Optional: so-vits-svc for singing voice synthesis
# See: https://github.com/svc-develop-team/so-vits-svc
```

## Usage

### Basic Usage

```python
from omg.lyrics_translation import LyricsTranslationPipeline

# Initialize pipeline
pipeline = LyricsTranslationPipeline(
    separator_model="htdemucs",
    aligner_model="MahmoudAshraf/mms-300m-1130-forced-aligner",
    output_dir="lyrics_translation_output",
)

# Run pipeline
results = pipeline.run(
    audio_path="path/to/original_song.wav",
    old_lyrics="Original lyrics here...",
    new_lyrics="New lyrics here...",
    output_name="my_cover",
)

# Access results
print(f"Cover song: {results['final_mix']}")
print(f"Vocals: {results['vocals']}")
print(f"Instrumental: {results['instrumental']}")
```

### Command Line Usage

```bash
# Basic usage
uv run python scripts/lyrics_translation_demo.py \
    --audio path/to/song.wav \
    --old-lyrics "Original lyrics" \
    --new-lyrics "New lyrics"

# Load lyrics from files
uv run python scripts/lyrics_translation_demo.py \
    --audio path/to/song.wav \
    --old-lyrics-file original_lyrics.txt \
    --new-lyrics-file new_lyrics.txt \
    --output-dir my_covers

# Specify models
uv run python scripts/lyrics_translation_demo.py \
    --audio path/to/song.wav \
    --old-lyrics-file original_lyrics.txt \
    --new-lyrics-file new_lyrics.txt \
    --separator-model htdemucs_ft \
    --device cuda
```

### Batch Processing

```python
from omg.lyrics_translation import LyricsTranslationPipeline

pipeline = LyricsTranslationPipeline()

# Process multiple songs
results = pipeline.run_batch(
    audio_paths=[
        "song1.wav",
        "song2.wav",
        "song3.wav",
    ],
    old_lyrics_list=[
        "Original lyrics 1...",
        "Original lyrics 2...",
        "Original lyrics 3...",
    ],
    new_lyrics_list=[
        "New lyrics 1...",
        "New lyrics 2...",
        "New lyrics 3...",
    ],
)
```

## Output Structure

The pipeline creates the following directory structure:

```
lyrics_translation_output/
└── song_name/
    ├── separated/
    │   └── htdemucs/
    │       └── song_name/
    │           ├── vocals.wav
    │           └── no_vocals.wav
    ├── alignment.txt
    ├── new_vocals.wav
    ├── song_name_cover.wav
    └── metadata.json
```

## API Reference

### LyricsTranslationPipeline

```python
class LyricsTranslationPipeline:
    def __init__(
        self,
        separator_model: str = "htdemucs",
        aligner_model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
        synthesizer_model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: Optional[str] = None,
        output_dir: str = "lyrics_translation_output",
    )
```

**Parameters:**
- `separator_model`: Demucs model name
- `aligner_model`: HuggingFace model for forced alignment
- `synthesizer_model`: TTS model name
- `device`: 'cuda' or 'cpu' (auto-detect if None)
- `output_dir`: Directory for outputs

**Methods:**

```python
def run(
    self,
    audio_path: str,
    old_lyrics: str,
    new_lyrics: str,
    output_name: Optional[str] = None,
    save_intermediate: bool = True,
) -> Dict[str, Any]:
    """Run the complete lyrics translation pipeline."""
```

**Returns:**
```python
{
    "vocals": str,          # Path to separated vocals
    "instrumental": str,    # Path to separated instrumental
    "alignment": str,       # Path to alignment file
    "new_vocals": str,      # Path to synthesized vocals
    "final_mix": str,       # Path to final cover song
    "metadata": str,        # Path to metadata JSON
    "word_timings": List[WordTiming],  # Timing data
}
```

## Components

### VocalSeparator

```python
from omg.lyrics_translation import VocalSeparator

separator = VocalSeparator(model_name="htdemucs")
vocals_path, instrumental_path = separator.separate(
    audio_path="song.wav",
    output_dir="separated",
)
```

### LyricsAligner

```python
from omg.lyrics_translation import LyricsAligner

aligner = LyricsAligner()
word_timings = aligner.align(
    audio_path="vocals.wav",
    lyrics="Lyrics text here...",
)

# Print alignment
aligner.print_alignment(word_timings)
```

### VoiceSynthesizer

```python
from omg.lyrics_translation import VoiceSynthesizer

synthesizer = VoiceSynthesizer()

# Synthesize new vocals
new_vocals = synthesizer.synthesize_from_lyrics(
    new_lyrics="New lyrics...",
    reference_vocals_path="vocals.wav",
    old_word_timings=word_timings,
    output_path="new_vocals.wav",
)

# Combine with instrumental
final = synthesizer.combine_with_instrumental(
    vocals_path=new_vocals,
    instrumental_path="instrumental.wav",
    output_path="cover.wav",
)
```

## Models Used

This module leverages the following HuggingFace models:

1. **Demucs** - Vocal separation
   - Paper: [Music Source Separation](https://arxiv.org/abs/2111.03600)
   - Models: `htdemucs`, `htdemucs_ft`, `mdx_extra`

2. **MMS Forced Aligner** - Lyrics alignment
   - Model: [MahmoudAshraf/mms-300m-1130-forced-aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner)
   - Based on: wav2vec2 with CTC alignment

3. **Voice Synthesis** - TTS + Voice Cloning
   - Recommended: seed-vc, so-vits-svc, OpenVoice, XTTS

## Current Limitations

1. **Voice Synthesis:** The current implementation uses a simplified placeholder. For production-quality singing:
   - Use [seed-vc](https://github.com/Plachtaa/seed-vc) for singing voice conversion
   - Use [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) for singing voice synthesis
   - Use [OpenVoice](https://github.com/myshell-ai/OpenVoice) for voice cloning

2. **Language Support:** Currently optimized for English. For other languages:
   - Set `language` parameter in LyricsAligner
   - Use appropriate MMS model for the target language

3. **Quality Factors:**
   - Clean vocals separation improves alignment quality
   - Clear pronunciation in reference audio improves synthesis
   - Similar rhythm between old and new lyrics produces better results

## Future Improvements

- [ ] Integrate seed-vc for singing voice conversion
- [ ] Add pitch extraction and manipulation
- [ ] Support multi-language lyrics
- [ ] Add prosody transfer from original vocals
- [ ] Implement real-time processing
- [ ] Add GUI interface
- [ ] Support harmony generation

## References

- [Demucs: Music Source Separation](https://github.com/facebookresearch/demucs)
- [CTC Forced Aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
- [Seed-VC: Singing Voice Conversion](https://github.com/Plachtaa/seed-vc)
- [So-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc)
- [OpenVoice: Voice Cloning](https://github.com/myshell-ai/OpenVoice)

## Citation

```bibtex
@misc{omg-lyrics-translation,
    title={Lyrics Translation and Cover Song Generation},
    author={Guan-Ming Chiu},
    year={2025},
    url={https://github.com/guan404ming/omg}
}
```
