"""
Constraint-based lyrics translation graph following ReAct pattern
"""

import logging
from typing import TypedDict, Optional, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from .tools import create_translation_tools

logger = logging.getLogger(__name__)


class LyricsTranslationState(TypedDict):
    """State for constraint-based lyrics translation graph"""

    # Input
    source_lyrics: str
    source_lang: str
    target_lang: str

    # Constraints
    constraints: Optional[dict]  # MusicConstraints as dict
    syllable_counts: Optional[list[int]]
    rhyme_scheme: Optional[str]
    syllable_patterns: Optional[list[str]]

    # Translation
    translated_lines: Optional[list[str]]
    reasoning: Optional[str]

    # Metrics
    translation_syllable_counts: Optional[list[int]]
    translation_rhyme_endings: Optional[list[str]]
    translation_syllable_patterns: Optional[list[str]]

    # Validation
    validation_passed: Optional[bool]
    validation_score: Optional[float]

    # Control
    attempt: int
    max_attempts: int
    all_lines_done: Optional[bool]
    messages: Annotated[list, add]


def create_lyrics_translation_initial_state(
    source_lyrics: str,
    source_lang: str,
    target_lang: str,
    constraints,
) -> LyricsTranslationState:
    """
    Create initial state for lyrics translation graph

    Args:
        source_lyrics: Source lyrics text
        source_lang: Source language code
        target_lang: Target language code
        constraints: MusicConstraints object with syllable_counts, rhyme_scheme, syllable_patterns

    Returns:
        LyricsTranslationState initialized with all required fields
    """
    return {
        "source_lyrics": source_lyrics,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "constraints": None,  # Not used directly in graph
        "syllable_counts": constraints.syllable_counts,
        "rhyme_scheme": constraints.rhyme_scheme,
        "syllable_patterns": constraints.syllable_patterns,
        "translated_lines": None,
        "reasoning": None,
        "translation_syllable_counts": None,
        "translation_rhyme_endings": None,
        "translation_syllable_patterns": None,
        "validation_passed": None,
        "validation_score": None,
        "attempt": 1,
        "max_attempts": 3,
        "all_lines_done": None,
        "messages": [],
    }


def build_lyrics_translation_graph(analyzer, llm, config):
    """
    Build the constraint-based lyrics translation graph using ReAct pattern.

    The graph uses a reasoning-acting cycle where the LLM reasons about the
    translation, uses tools to verify syllable counts, and iterates until
    all constraints are satisfied.

    Args:
        analyzer: LyricsAnalyzer instance
        llm: LLM instance (ChatOllama)
        config: LyricsTranslationAgentConfig instance

    Returns:
        Compiled LangGraph workflow
    """
    # Create and bind tools
    tools = create_translation_tools(analyzer)
    llm.bind_tools(tools)

    def translate_line_node(state: LyricsTranslationState) -> dict:
        """Translate one line at a time with iterative verification"""
        source_lines = [
            line.strip()
            for line in state["source_lyrics"].strip().split("\n")
            if line.strip()
        ]
        target_syllables = state["syllable_counts"] or []
        translated_lines = list(state.get("translated_lines") or [])
        current_idx = len(translated_lines)
        target_lang = state["target_lang"]

        if current_idx >= len(source_lines):
            return {"translated_lines": translated_lines}

        source_line = source_lines[current_idx]
        target_count = (
            target_syllables[current_idx] if current_idx < len(target_syllables) else 0
        )

        logger.info(
            f'   🚀 Line {current_idx + 1}/{len(source_lines)}: "{source_line}" → {target_count} syllables'
        )

        # Simple iterative approach - translate and verify
        max_attempts = 5
        best_translation = ""
        best_diff = float("inf")

        for attempt in range(max_attempts):
            # Build prompt with feedback
            if attempt == 0:
                prompt = f"""Translate to {target_lang} with EXACTLY {target_count} syllables:

"{source_line}"

Output ONLY the translation, nothing else."""
            else:
                actual = analyzer.count_syllables(best_translation, target_lang)
                diff = actual - target_count
                if diff > 0:
                    feedback = (
                        f"Too long ({actual} syllables). Remove {diff} syllable(s)."
                    )
                else:
                    feedback = (
                        f"Too short ({actual} syllables). Add {abs(diff)} syllable(s)."
                    )

                prompt = f"""Your translation "{best_translation}" has {actual} syllables but needs {target_count}.
{feedback}

Revise to have EXACTLY {target_count} syllables. Output ONLY the translation."""

            response = llm.invoke(
                [
                    {"role": "system", "content": config.get_system_prompt()},
                    {"role": "user", "content": prompt},
                ]
            )

            # Extract translation from response
            translation = response.content.strip()
            # Clean up common formatting issues
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1]
            if translation.startswith("```") or translation.startswith("{"):
                # Try to extract just the text
                import re

                match = re.search(r'"([^"]+)"', translation)
                if match:
                    translation = match.group(1)
                else:
                    # Just take the first line that looks like text
                    for line in translation.split("\n"):
                        line = line.strip()
                        if (
                            line
                            and not line.startswith("{")
                            and not line.startswith("```")
                        ):
                            translation = line
                            break

            # Check syllable count
            actual = analyzer.count_syllables(translation, target_lang)
            diff = abs(actual - target_count)

            if diff < best_diff:
                best_diff = diff
                best_translation = translation

            if actual == target_count:
                logger.info(
                    f'      ✓ Attempt {attempt + 1}: "{translation}" ({actual}/{target_count})'
                )
                break
            else:
                logger.info(
                    f'      ⚠ Attempt {attempt + 1}: "{translation}" ({actual}/{target_count})'
                )

        translated_lines.append(best_translation)
        return {"translated_lines": translated_lines}

    def check_progress_node(state: LyricsTranslationState) -> dict:
        """Check if all lines are translated"""
        source_lines = [
            line.strip()
            for line in state["source_lyrics"].strip().split("\n")
            if line.strip()
        ]
        translated_lines = state.get("translated_lines") or []
        return {"all_lines_done": len(translated_lines) >= len(source_lines)}

    def calculate_metrics_node(state: LyricsTranslationState) -> dict:
        """Calculate final translation metrics"""
        if not state.get("translated_lines"):
            return {
                "translation_syllable_counts": [],
                "translation_rhyme_endings": [],
                "translation_syllable_patterns": [],
            }

        translated_lines = state["translated_lines"]
        target_lang = state["target_lang"]

        syllable_counts = [
            analyzer.count_syllables(line, target_lang) for line in translated_lines
        ]
        rhyme_endings = [
            analyzer.extract_rhyme_ending(line, target_lang)
            for line in translated_lines
        ]
        syllable_patterns = analyzer.get_syllable_patterns(
            translated_lines, target_lang
        )

        # Print summary
        expected = state.get("syllable_counts") or []
        matches = sum(1 for exp, act in zip(expected, syllable_counts) if exp == act)
        score = matches / len(expected) if expected else 0.0
        print(
            f"\n   {'✓' if score >= 0.75 else '⚠'} Final: {matches}/{len(expected)} lines match ({score:.0%})"
        )

        return {
            "translation_syllable_counts": syllable_counts,
            "translation_rhyme_endings": rhyme_endings,
            "translation_syllable_patterns": syllable_patterns,
        }

    def should_continue(state: LyricsTranslationState) -> str:
        """Decide whether to translate next line or finish"""
        if state.get("all_lines_done"):
            return "calculate"
        return "translate"

    # Build workflow
    workflow = StateGraph(LyricsTranslationState)

    workflow.add_node("translate_line", translate_line_node)
    workflow.add_node("check_progress", check_progress_node)
    workflow.add_node("calculate_metrics", calculate_metrics_node)

    workflow.set_entry_point("translate_line")

    workflow.add_edge("translate_line", "check_progress")
    workflow.add_conditional_edges(
        "check_progress",
        should_continue,
        {"translate": "translate_line", "calculate": "calculate_metrics"},
    )
    workflow.add_edge("calculate_metrics", END)

    return workflow.compile()
