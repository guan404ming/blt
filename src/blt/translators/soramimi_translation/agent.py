"""
Soramimi Translation Agent

Creates phonetically similar text through phoneme mapping (空耳).
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv

from langchain_ollama import ChatOllama

from ..shared import SoramimiTranslation, LyricsAnalyzer
from .config import SoramimiTranslationAgentConfig
from .graph import build_soramimi_mapping_graph, create_soramimi_mapping_initial_state
from .validator import SoramimiValidator

load_dotenv()
logger = logging.getLogger(__name__)


class SoramimiTranslationAgent:
    """Soramimi translator using phoneme mapping"""

    def __init__(
        self,
        config: Optional[SoramimiTranslationAgentConfig] = None,
        analyzer: Optional[LyricsAnalyzer] = None,
    ):
        """Initialize soramimi translator"""
        self.config = config or SoramimiTranslationAgentConfig()
        self.analyzer = analyzer or LyricsAnalyzer()
        self.validator = SoramimiValidator(
            self.analyzer,
            self.config.similarity_threshold,
        )

        # Configure LangSmith
        if self.config.langsmith_tracing:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.config.langsmith_project
            if not os.getenv("LANGCHAIN_API_KEY"):
                logger.warning(
                    "LangSmith tracing enabled but LANGCHAIN_API_KEY not set."
                )

        # Create LLM
        self.llm = ChatOllama(
            model=self.config.model,
            base_url=self.config.ollama_base_url.replace("/v1", ""),
            format="json",
            temperature=0.7,
        )

        # Build graph
        self.graph = build_soramimi_mapping_graph(
            self.analyzer, self.validator, self.llm
        )

    def translate(
        self,
        source_lyrics: Union[str, list[str]],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> SoramimiTranslation:
        """
        Create soramimi translation using phoneme mapping

        Args:
            source_lyrics: Source lyrics (string or list of lines)
            source_lang: Source language
            target_lang: Target language

        Returns:
            SoramimiTranslation
        """
        start_time = time.time()

        source_lang = source_lang or self.config.default_source_lang
        target_lang = target_lang or self.config.default_target_lang

        # Parse lines
        if isinstance(source_lyrics, str):
            source_lines = [
                line.strip()
                for line in source_lyrics.strip().split("\n")
                if line.strip()
            ]
        else:
            source_lines = [line.strip() for line in source_lyrics if line.strip()]

        # Handle Chinese -> Pinyin
        chinese_lang_codes = ["cmn", "zh", "zh-cn", "zh-tw"]
        if source_lang.lower() in chinese_lang_codes:
            logger.info("   Converting Chinese to pinyin")
            source_lines = [
                self.analyzer._chinese_to_pinyin(line) for line in source_lines
            ]
            source_lang = "en-us"

        # Early return if same language
        if source_lang == target_lang:
            return SoramimiTranslation(
                soramimi_lines=source_lines,
                source_ipa=[],
                target_ipa=[],
                similarity_scores=[1.0] * len(source_lines),
                overall_similarity=1.0,
                reasoning="Same language",
            )

        # Initialize state
        initial_state = create_soramimi_mapping_initial_state(
            source_lines,
            source_lang,
            target_lang,
            self.config.max_retries,
            self.config.similarity_threshold,
        )

        # Run graph
        print("   🗺️  Starting mapping-based soramimi creation...")
        final_state = self.graph.invoke(
            initial_state,
            config={
                "recursion_limit": 100,
                "run_name": "SoramimiTranslation",
            },
        )

        # Build result
        best_lines = final_state["best_lines"]
        best_scores = final_state["best_scores"]
        best_ipas = final_state["best_ipas"]

        if not best_lines or any(line is None for line in best_lines):
            raise RuntimeError("Failed to generate soramimi translation")

        overall_similarity = sum(best_scores) / len(best_scores) if best_scores else 0

        translation = SoramimiTranslation(
            soramimi_lines=list(best_lines),
            source_ipa=[ipa[0] for ipa in best_ipas],
            target_ipa=[ipa[1] for ipa in best_ipas],
            similarity_scores=list(best_scores),
            overall_similarity=overall_similarity,
            reasoning=f"Mapping-based: {len(final_state['phoneme_mapping'])} phonemes mapped over {final_state['attempt']} attempts",
        )

        # Display
        elapsed = time.time() - start_time
        print(
            f"\n   ✓ Completed in {elapsed:.1f}s - Similarity: {overall_similarity:.1%}"
        )

        # Auto-save
        if self.config.auto_save:
            save_dir = Path(self.config.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"soramimi_mapping_{source_lang}_{target_lang}_{timestamp}.{self.config.save_format}"
            file_path = save_dir / filename
            translation.save(str(file_path), format=self.config.save_format)
            logger.info(f"Soramimi translation saved to {file_path}")

        return translation
