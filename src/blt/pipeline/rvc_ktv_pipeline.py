"""Complete pipeline for RVC voice conversion and KTV video generation.

This pipeline orchestrates the entire process:
1. Vocal separation (separate vocals from instrumental)
2. RVC voice conversion (convert vocals to target voice)
3. Audio mixing (combine converted vocals with instrumental)
4. Lip-sync video generation (create lip-synced video)
5. KTV video generation (add dual-track subtitles)
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json

from blt.synthesizer import (
    VocalSeparator,
    RetrievalBasedVoiceConverter,
    AudioMixer,
    WhisperLyricsAligner,
    LipSyncedVideoGenerator,
    KTVVideoGenerator,
)


class RVCKTVPipeline:
    """End-to-end pipeline for RVC voice conversion and KTV video generation.

    This pipeline takes:
    - Original song audio
    - RVC model files (model.pth, model.index)
    - Face image/video for lip-sync
    - Lyrics (main and optional sub-lyrics)

    And produces:
    - Voice-converted audio
    - Lip-synced video
    - KTV video with dual-track subtitles

    Args:
        separator_model: Demucs model for vocal separation
        whisper_model: Whisper model size for lyrics alignment
        device: Device to run models on ('cuda' or 'cpu')
        output_dir: Directory to save all outputs
    """

    def __init__(
        self,
        separator_model: str = "htdemucs",
        whisper_model: str = "medium",
        device: Optional[str] = None,
        output_dir: str = "rvc_ktv_output",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("INITIALIZING RVC KTV PIPELINE")
        print("=" * 60)

        # Initialize components
        self.separator = VocalSeparator(
            model_name=separator_model,
            device=device,
        )

        self.rvc_converter = RetrievalBasedVoiceConverter()

        self.audio_mixer = AudioMixer(normalize=True)

        self.lyrics_aligner = WhisperLyricsAligner(
            model_size=whisper_model,
            device=device,
        )

        self.lipsync_generator = LipSyncedVideoGenerator()

        self.ktv_generator = KTVVideoGenerator()

        print("\n✓ All components initialized successfully!")

    def run(
        self,
        audio_path: str,
        rvc_model_path: str,
        rvc_index_path: str,
        face_path: str,
        main_lyrics: str,
        sub_lyrics: Optional[str] = None,
        lyrics_language: str = "zh",
        output_name: Optional[str] = None,
        pitch_shift: int = 0,
        vocal_volume: float = 1.0,
        instrumental_volume: float = 0.8,
    ) -> Dict[str, Any]:
        """Run the complete RVC KTV pipeline.

        Args:
            audio_path: Path to original song audio
            rvc_model_path: Path to RVC model (.pth file)
            rvc_index_path: Path to RVC index (.index file)
            face_path: Path to face image/video for lip-sync
            main_lyrics: Main lyrics text (newline-separated)
            sub_lyrics: Optional sub-lyrics (e.g., phonetic translation)
            lyrics_language: Language code for lyrics (default: 'zh' for Chinese)
            output_name: Name for output files (default: uses input filename)
            pitch_shift: Pitch shift (male to female +12, female to male -12, default: 0)
            vocal_volume: Volume multiplier for vocals (default: 1.0)
            instrumental_volume: Volume multiplier for instrumental (default: 0.8)

        Returns:
            Dictionary with paths to all outputs
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
        print(f"RVC model: {rvc_model_path}")
        print(f"Face: {face_path}")
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

        # Step 2: RVC voice conversion
        print("\n" + "=" * 60)
        print("STEP 2: RVC VOICE CONVERSION")
        print("=" * 60)

        converted_vocals_path = song_dir / f"{output_name}_converted_vocals.wav"
        self.rvc_converter.run(
            audio_path=vocals_path,
            model_path=rvc_model_path,
            index_path=rvc_index_path,
            output_path=str(converted_vocals_path),
            pitch_shift=pitch_shift,
        )

        results["converted_vocals"] = str(converted_vocals_path)

        # Step 3: Mix converted vocals with instrumental
        print("\n" + "=" * 60)
        print("STEP 3: AUDIO MIXING")
        print("=" * 60)

        mixed_audio_path = song_dir / f"{output_name}_mixed.wav"
        self.audio_mixer.mix(
            audio1_path=converted_vocals_path,
            audio2_path=instrumental_path,
            output_path=str(mixed_audio_path),
            volume1=vocal_volume,
            volume2=instrumental_volume,
        )

        results["mixed_audio"] = str(mixed_audio_path)

        # Step 4: Generate lip-synced video
        print("\n" + "=" * 60)
        print("STEP 4: LIP-SYNC VIDEO GENERATION")
        print("=" * 60)

        lipsync_video_path = song_dir / f"{output_name}_lipsync.mp4"
        self.lipsync_generator.generate(
            face_path=face_path,
            audio_path=str(mixed_audio_path),
            output_path=str(lipsync_video_path),
            resize_factor=1,
        )

        results["lipsync_video"] = str(lipsync_video_path)

        # Step 5: Align lyrics
        print("\n" + "=" * 60)
        print("STEP 5: LYRICS ALIGNMENT")
        print("=" * 60)

        alignment_result = self.lyrics_aligner.align(
            audio_path=str(mixed_audio_path),
            lyrics=main_lyrics,
            language=lyrics_language,
        )

        results["alignment"] = alignment_result

        # Step 6: Generate KTV video with subtitles
        print("\n" + "=" * 60)
        print("STEP 6: KTV VIDEO GENERATION")
        print("=" * 60)

        ktv_video_path = song_dir / f"{output_name}_ktv.mp4"
        self.ktv_generator.generate(
            video_path=str(lipsync_video_path),
            audio_path=str(mixed_audio_path),
            main_lyrics=main_lyrics,
            alignment_result=alignment_result,
            output_path=str(ktv_video_path),
            sub_lyrics=sub_lyrics,
        )

        results["ktv_video"] = str(ktv_video_path)

        # Save metadata
        metadata = {
            "input_audio": str(audio_path),
            "rvc_model": rvc_model_path,
            "rvc_index": rvc_index_path,
            "face": face_path,
            "main_lyrics": main_lyrics,
            "sub_lyrics": sub_lyrics,
            "lyrics_language": lyrics_language,
            "output_name": output_name,
            "pitch_shift": pitch_shift,
            "volumes": {
                "vocal": vocal_volume,
                "instrumental": instrumental_volume,
            },
            "outputs": {
                "vocals": results["vocals"],
                "instrumental": results["instrumental"],
                "converted_vocals": results["converted_vocals"],
                "mixed_audio": results["mixed_audio"],
                "lipsync_video": results["lipsync_video"],
                "ktv_video": results["ktv_video"],
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
        print(f"Mixed audio: {results['mixed_audio']}")
        print(f"Lip-sync video: {results['lipsync_video']}")
        print(f"KTV video: {results['ktv_video']}")
        print(f"Metadata: {results['metadata']}")
        print("=" * 60)

        return results
