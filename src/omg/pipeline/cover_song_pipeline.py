"""Complete pipeline for lyrics translation and cover song generation.

This pipeline orchestrates the entire process:
1. Vocal separation (separate vocals from instrumental)
2. Lyrics alignment (align old lyrics with vocals)
3. Voice synthesis (generate new vocals with new lyrics)
4. Mixing (combine new vocals with original instrumental)
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json

from omg.synthesizer.vocal_separator import VocalSeparator
from omg.synthesizer.lyrics_aligner import LyricsAligner
from omg.synthesizer.voice_synthesizer import VoiceSynthesizer


class CoverSongPipeline:
    """End-to-end pipeline for generating cover songs with new lyrics.

    This pipeline takes:
    - Original song audio
    - Original lyrics
    - New lyrics

    And produces:
    - Cover song with new lyrics in original singer's voice style

    Args:
        separator_model: Demucs model for vocal separation
        aligner_model: Model for lyrics alignment
        synthesizer_model: Model for voice synthesis
        device: Device to run models on ('cuda' or 'cpu')
        output_dir: Directory to save all outputs
    """

    def __init__(
        self,
        separator_model: str = "htdemucs",
        aligner_model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
        synthesizer_model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: Optional[str] = None,
        output_dir: str = "lyrics_translation_output",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("INITIALIZING COVER SONG PIPELINE")
        print("=" * 60)

        # Initialize components
        self.separator = VocalSeparator(
            model_name=separator_model,
            device=device,
        )

        self.aligner = LyricsAligner(
            model_name=aligner_model,
            device=device,
        )

        self.synthesizer = VoiceSynthesizer(
            model_name=synthesizer_model,
            device=device,
        )

        print("\n✓ All components initialized successfully!")

    def run(
        self,
        audio_path: str,
        old_lyrics: str,
        new_lyrics: str,
        output_name: Optional[str] = None,
        save_intermediate: bool = True,
    ) -> Dict[str, Any]:
        """Run the complete lyrics translation pipeline.

        Args:
            audio_path: Path to original song audio
            old_lyrics: Original lyrics text
            new_lyrics: New lyrics to synthesize
            output_name: Name for output files (default: uses input filename)
            save_intermediate: Whether to save intermediate outputs

        Returns:
            Dictionary with paths to all outputs:
            - vocals: Path to separated vocals
            - instrumental: Path to separated instrumental
            - alignment: Path to lyrics alignment file
            - new_vocals: Path to synthesized vocals with new lyrics
            - final_mix: Path to final mixed audio
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if output_name is None:
            output_name = audio_path.stem

        # Create output directory for this song
        song_dir = self.output_dir / output_name
        song_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"PROCESSING: {output_name}")
        print("=" * 60)
        print(f"Input audio: {audio_path}")
        print(f"Old lyrics: {old_lyrics[:50]}...")
        print(f"New lyrics: {new_lyrics[:50]}...")
        print(f"Output directory: {song_dir}")

        results = {}

        # Step 1: Separate vocals from instrumental
        print("\n" + "=" * 60)
        print("STEP 1: VOCAL SEPARATION")
        print("=" * 60)

        vocals_path, instrumental_path = self.separator.separate(
            audio_path=str(audio_path),
            output_dir=str(song_dir / "separated"),
        )

        results["vocals"] = vocals_path
        results["instrumental"] = instrumental_path

        # Step 2: Align old lyrics with vocals
        print("\n" + "=" * 60)
        print("STEP 2: LYRICS ALIGNMENT")
        print("=" * 60)

        word_timings = self.aligner.align(
            audio_path=vocals_path,
            lyrics=old_lyrics,
        )

        # Save alignment
        alignment_path = song_dir / "alignment.txt"
        self.aligner.save_alignment(word_timings, str(alignment_path))
        self.aligner.print_alignment(word_timings, max_words=10)

        results["alignment"] = str(alignment_path)
        results["word_timings"] = word_timings

        # Map new lyrics to old lyrics timing
        print("\n" + "=" * 60)
        print("STEP 2B: MAPPING NEW LYRICS TO ALIGNMENT")
        print("=" * 60)

        new_word_timings = self.aligner.map_new_lyrics_to_timing(
            old_word_timings=word_timings,
            new_lyrics=new_lyrics,
        )

        # Print new lyrics alignment (optional - for debugging)
        self.aligner.print_alignment(new_word_timings, max_words=0)

        results["new_word_timings"] = new_word_timings

        # Step 3: Synthesize new vocals
        print("\n" + "=" * 60)
        print("STEP 3: VOICE SYNTHESIS")
        print("=" * 60)

        new_vocals_path = song_dir / "new_vocals.wav"
        synthesized = self.synthesizer.synthesize_from_lyrics(
            new_lyrics=new_lyrics,
            reference_vocals_path=vocals_path,
            old_word_timings=word_timings,
            output_path=str(new_vocals_path),
        )

        results["new_vocals"] = synthesized

        # Step 4: Mix new vocals with original instrumental
        print("\n" + "=" * 60)
        print("STEP 4: FINAL MIXING")
        print("=" * 60)

        final_path = song_dir / f"{output_name}_cover.wav"
        mixed = self.synthesizer.combine_with_instrumental(
            vocals_path=synthesized,
            instrumental_path=instrumental_path,
            output_path=str(final_path),
            vocals_gain=1.0,
            instrumental_gain=0.8,
        )

        results["final_mix"] = mixed

        # Save metadata
        metadata = {
            "input_audio": str(audio_path),
            "old_lyrics": old_lyrics,
            "new_lyrics": new_lyrics,
            "output_name": output_name,
            "outputs": {
                "vocals": results["vocals"],
                "instrumental": results["instrumental"],
                "alignment": results["alignment"],
                "new_vocals": results["new_vocals"],
                "final_mix": results["final_mix"],
            },
        }

        metadata_path = song_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        results["metadata"] = str(metadata_path)

        # Print summary
        print("\n" + "=" * 60)
        print("✓ PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"Final cover song: {results['final_mix']}")
        print(f"Metadata: {results['metadata']}")
        print("=" * 60)

        return results

    def run_batch(
        self,
        audio_paths: list[str],
        old_lyrics_list: list[str],
        new_lyrics_list: list[str],
    ) -> list[Dict[str, Any]]:
        """Run pipeline on multiple songs.

        Args:
            audio_paths: List of audio file paths
            old_lyrics_list: List of original lyrics
            new_lyrics_list: List of new lyrics

        Returns:
            List of result dictionaries for each song
        """
        if not (len(audio_paths) == len(old_lyrics_list) == len(new_lyrics_list)):
            raise ValueError("All input lists must have the same length")

        results = []
        for i, (audio, old_lyrics, new_lyrics) in enumerate(
            zip(audio_paths, old_lyrics_list, new_lyrics_list)
        ):
            print(f"\n\n{'=' * 60}")
            print(f"PROCESSING SONG {i + 1}/{len(audio_paths)}")
            print(f"{'=' * 60}\n")

            result = self.run(
                audio_path=audio,
                old_lyrics=old_lyrics,
                new_lyrics=new_lyrics,
            )
            results.append(result)

        print("\n\n" + "=" * 60)
        print(f"✓ BATCH PROCESSING COMPLETE: {len(results)} songs")
        print("=" * 60)

        return results
