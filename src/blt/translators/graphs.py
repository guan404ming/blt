"""
LangGraph state and graph definitions for soramimi translation
"""

import json
import logging
from pathlib import Path
from typing import TypedDict, Optional, Annotated
from operator import add

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

# Directory for storing fallback mappings
FALLBACK_MAPPINGS_DIR = Path(__file__).parent / "fallback_mappings"


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


class SoramimiMappingState(TypedDict):
    """State for mapping-based soramimi translation graph"""

    # Source information
    source_lines: list[str]
    source_lang: str
    target_lang: str

    # Phoneme mapping
    source_phonemes: list[str]  # Unique phonemes from source
    phoneme_mapping: dict[str, str]  # phoneme -> target character/syllable
    mapping_scores: dict[str, float]  # phoneme -> similarity score

    # Current translation
    soramimi_lines: Optional[list[str]]
    source_ipa: Optional[list[str]]
    target_ipa: Optional[list[str]]
    similarity_scores: Optional[list[float]]
    overall_similarity: Optional[float]

    # Best results
    best_mapping: Optional[dict[str, str]]
    best_lines: Optional[list[str]]
    best_scores: Optional[list[float]]
    best_ipas: Optional[list[tuple[str, str]]]

    # Control
    attempt: int
    max_attempts: int
    threshold: float
    messages: Annotated[list, add]


def _get_fallback_file_path(target_lang: str) -> Path:
    """Get the file path for a language's fallback mapping"""
    # Normalize language code
    lang_code = target_lang.lower().split("-")[0]  # e.g., "zh-cn" -> "zh"

    # Map common variants to canonical names
    lang_map = {
        "cmn": "zh",
        "zh-cn": "zh",
        "zh-tw": "zh",
    }
    lang_code = lang_map.get(lang_code, lang_code)

    return FALLBACK_MAPPINGS_DIR / f"{lang_code}.json"


def _load_fallback_mapping(target_lang: str) -> dict[str, str]:
    """
    Load fallback mapping from JSON file

    Args:
        target_lang: Target language code

    Returns:
        Fallback mapping dict (falls back to en.json if language not found)
    """
    file_path = _get_fallback_file_path(target_lang)

    # Try language-specific file first
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                logger.info(f"   Loaded fallback mapping from {file_path.name}")
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load fallback mapping from {file_path}: {e}")

    # Fall back to default English mapping
    default_path = FALLBACK_MAPPINGS_DIR / "en.json"
    if default_path.exists():
        try:
            with open(default_path, "r", encoding="utf-8") as f:
                logger.info("   Using default fallback mapping (en.json)")
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load default fallback mapping: {e}")

    # Last resort: return empty dict
    logger.warning(f"No fallback mapping found for {target_lang}, using empty mapping")
    return {}


def build_soramimi_mapping_graph(analyzer, validator, llm):
    """
    Build the mapping-based soramimi translation graph

    Args:
        analyzer: LyricsAnalyzer instance
        validator: SoramimiValidator instance
        llm: LLM instance (ChatOllama)

    Returns:
        Compiled LangGraph
    """

    # Cache for fallback mappings per language
    fallback_cache = {}

    def get_fallback_mapping(target_lang: str) -> dict[str, str]:
        """Get or load fallback mapping for target language"""
        if target_lang not in fallback_cache:
            fallback_cache[target_lang] = _load_fallback_mapping(target_lang)
        return fallback_cache[target_lang]

    def extract_phonemes_node(state: SoramimiMappingState) -> dict:
        """Extract unique phonemes from source lyrics"""

        print("   🔍 Extracting phonemes from source...")

        # Get IPA for all source lines
        source_ipas = [
            analyzer.text_to_ipa(line, state["source_lang"])
            for line in state["source_lines"]
        ]

        # Extract unique phonemes (split by space)
        all_phonemes = set()
        for ipa in source_ipas:
            # Split IPA into individual phonemes
            phonemes = ipa.split()
            all_phonemes.update(phonemes)

        source_phonemes = sorted(list(all_phonemes))
        print(f"   📝 Found {len(source_phonemes)} unique phonemes")

        return {
            "source_phonemes": source_phonemes,
            "source_ipa": source_ipas,
        }

    def build_mapping_node(state: SoramimiMappingState) -> dict:
        """Build or refine phoneme mapping"""

        attempt = state["attempt"]
        print(f"   🔄 Building mapping (Attempt {attempt}/{state['max_attempts']})")

        if attempt == 1:
            # Initial mapping
            prompt = get_initial_mapping_prompt(state)
        else:
            # Refine existing mapping
            prompt = get_refinement_prompt(state)

        # Call LLM
        response = llm.invoke([{"role": "user", "content": prompt}])

        try:
            result = json.loads(response.content)
            phoneme_mapping = result.get("phoneme_mapping", {})

            # Merge with existing mapping to keep previous mappings
            existing_mapping = state.get("phoneme_mapping", {})
            merged_mapping = {**existing_mapping, **phoneme_mapping}

            # Check for unmapped phonemes
            unmapped = [p for p in state["source_phonemes"] if p not in merged_mapping]

            if unmapped:
                print(
                    f"   ⚠️  {len(unmapped)} phonemes still unmapped: {unmapped[:5]}..."
                )
            else:
                print(f"   ✓ All {len(merged_mapping)} phonemes mapped!")

            return {
                "phoneme_mapping": merged_mapping,
                "messages": state.get("messages", [])
                + [{"role": "assistant", "content": response.content}],
            }
        except json.JSONDecodeError as e:
            logger.warning(f"   Failed to parse mapping: {e}")
            return {
                "phoneme_mapping": state.get("phoneme_mapping", {}),
                "messages": state.get("messages", [])
                + [{"role": "assistant", "content": response.content}],
            }

    def apply_mapping_node(state: SoramimiMappingState) -> dict:
        """Apply phoneme mapping to generate soramimi"""

        print("   🎨 Applying mapping to generate soramimi...")

        mapping = state["phoneme_mapping"]
        fallback_mapping = get_fallback_mapping(state["target_lang"])
        soramimi_lines = []

        for source_ipa in state["source_ipa"]:
            # Split IPA into phonemes
            phonemes = source_ipa.split()

            # Map each phoneme
            mapped = []
            unmapped_count = 0
            for phoneme in phonemes:
                if phoneme in mapping:
                    mapped.append(mapping[phoneme])
                elif phoneme in fallback_mapping:
                    # Use cached fallback mapping
                    mapped.append(fallback_mapping[phoneme])
                    unmapped_count += 1
                else:
                    # Last resort: use placeholder
                    mapped.append("？")
                    unmapped_count += 1

            if unmapped_count > 0:
                logger.warning(f"   Line has {unmapped_count} unmapped phonemes")

            soramimi_line = "".join(mapped)
            soramimi_lines.append(soramimi_line)

        return {"soramimi_lines": soramimi_lines}

    def validate_node(state: SoramimiMappingState) -> dict:
        """Validate phonetic similarity"""

        if not state.get("soramimi_lines"):
            return {
                "similarity_scores": [0.0] * len(state["source_lines"]),
                "overall_similarity": 0.0,
            }

        validation = validator.compare_ipa(
            state["source_lines"],
            state["soramimi_lines"],
            state["source_lang"],
            state["target_lang"],
        )

        # Calculate per-phoneme scores
        mapping_scores = {}
        for phoneme in state["source_phonemes"]:
            # Calculate average score for this phoneme
            # (simplified - could be more sophisticated)
            mapping_scores[phoneme] = validation["overall_similarity"]

        # Update best results
        best_lines = state.get("best_lines") or [None] * len(state["source_lines"])
        best_scores = state.get("best_scores") or [0.0] * len(state["source_lines"])
        best_ipas = state.get("best_ipas") or [("", "")] * len(state["source_lines"])
        best_mapping = state.get("best_mapping")

        current_avg = validation["overall_similarity"]
        best_avg = sum(best_scores) / len(best_scores) if best_scores else 0

        if current_avg > best_avg:
            print(f"   ✓ Improved: {best_avg:.1%} → {current_avg:.1%}")
            best_lines = state["soramimi_lines"][:]
            best_scores = validation["similarities"][:]
            best_ipas = [
                (src, tgt)
                for src, tgt in zip(
                    validation["source_ipas"], validation["target_ipas"]
                )
            ]
            best_mapping = state["phoneme_mapping"].copy()

        print(f"   📊 Current best: {max(current_avg, best_avg):.1%}")

        return {
            "target_ipa": validation["target_ipas"],
            "similarity_scores": validation["similarities"],
            "overall_similarity": validation["overall_similarity"],
            "mapping_scores": mapping_scores,
            "best_lines": best_lines,
            "best_scores": best_scores,
            "best_ipas": best_ipas,
            "best_mapping": best_mapping,
        }

    def refine_mapping_node(state: SoramimiMappingState) -> dict:
        """Increment attempt counter"""
        return {"attempt": state["attempt"] + 1}

    def should_continue(state: SoramimiMappingState) -> str:
        """Decide whether to continue refining"""

        best_scores = state.get("best_scores", [])
        if not best_scores:
            return "refine"

        avg_score = sum(best_scores) / len(best_scores)

        if avg_score >= state["threshold"]:
            print("   ✅ Mapping meets threshold!")
            return "end"

        if state["attempt"] >= state["max_attempts"]:
            print(f"   ⚠️  Max attempts ({state['max_attempts']}) reached")
            return "end"

        print("   🔁 Refining mapping...")
        return "refine"

    def get_initial_mapping_prompt(state: SoramimiMappingState) -> str:
        """Generate initial mapping prompt"""

        # Show all phonemes
        phonemes_list = ", ".join(f'"{p}"' for p in state["source_phonemes"])

        return f"""Create a COMPLETE phoneme mapping for soramimi (空耳) translation.

Source language: {state["source_lang"]}
Target language: {state["target_lang"]}

ALL {len(state["source_phonemes"])} source phonemes (IPA):
{phonemes_list}

For EVERY SINGLE phoneme above, find a {state["target_lang"]} character that sounds similar.

Common IPA to Chinese mappings:
- Vowels: "ɪ"→"伊", "ɛ"→"埃", "æ"→"啊", "ʌ"→"啊", "ɔ"→"奥", "ʊ"→"乌", "ə"→"额"
- Consonants: "h"→"赫", "l"→"勒", "r"→"尔", "w"→"瓦", "n"→"恩", "m"→"姆", "s"→"斯", "t"→"特", "d"→"德", "k"→"克", "g"→"格", "p"→"普", "b"→"布", "f"→"弗", "v"→"夫"
- Diphthongs: "eɪ"→"诶", "aɪ"→"艾", "ɔɪ"→"奥伊", "aʊ"→"奥", "oʊ"→"欧"
- Special: "ð"→"德", "θ"→"斯", "ʃ"→"施", "ʒ"→"日", "ŋ"→"嗯"

Return COMPLETE JSON mapping (MUST include all {len(state["source_phonemes"])} phonemes):
{{
  "phoneme_mapping": {{
    "<phoneme1>": "<chinese_char>",
    "<phoneme2>": "<chinese_char>",
    ...
  }}
}}

CRITICAL: Every phoneme from the list above MUST have a mapping. Missing even one phoneme is unacceptable!
"""

    def get_refinement_prompt(state: SoramimiMappingState) -> str:
        """Generate refinement prompt"""

        current_mapping = state["phoneme_mapping"]
        unmapped = [p for p in state["source_phonemes"] if p not in current_mapping]

        if unmapped:
            # Focus on mapping the unmapped phonemes first
            unmapped_list = ", ".join(f'"{p}"' for p in unmapped)
            return f"""URGENT: Complete the phoneme mapping!

These {len(unmapped)} phonemes are still UNMAPPED:
{unmapped_list}

Map EACH of these to a Chinese character with similar sound.

Return JSON with mappings for these unmapped phonemes:
{{
  "phoneme_mapping": {{
    "<unmapped_phoneme1>": "<chinese_char>",
    "<unmapped_phoneme2>": "<chinese_char>",
    ...
  }}
}}

CRITICAL: Must map ALL {len(unmapped)} unmapped phonemes!
"""
        else:
            # All mapped, refine low-scoring ones
            poor_phonemes = [
                (p, state["mapping_scores"].get(p, 0))
                for p in state["source_phonemes"]
                if state["mapping_scores"].get(p, 0) < state["threshold"]
            ][:10]

            poor_list = "\n".join(
                f"- {p} -> {current_mapping.get(p, '?')} (score: {score:.1%})"
                for p, score in poor_phonemes
            )

            return f"""Refine phoneme mappings to improve similarity.

Phonemes below threshold ({state["threshold"]:.0%}):
{poor_list}

Improve these by finding better Chinese character matches.

Return JSON with improved mappings:
{{
  "phoneme_mapping": {{ ... }}
}}
"""

    # Build workflow
    workflow = StateGraph(SoramimiMappingState)

    # Add nodes
    workflow.add_node("extract_phonemes", extract_phonemes_node)
    workflow.add_node("build_mapping", build_mapping_node)
    workflow.add_node("apply_mapping", apply_mapping_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("refine_mapping", refine_mapping_node)

    # Set entry point
    workflow.set_entry_point("extract_phonemes")

    # Add edges
    workflow.add_edge("extract_phonemes", "build_mapping")
    workflow.add_edge("build_mapping", "apply_mapping")
    workflow.add_edge("apply_mapping", "validate")
    workflow.add_conditional_edges(
        "validate",
        should_continue,
        {
            "refine": "refine_mapping",
            "end": END,
        },
    )
    workflow.add_edge("refine_mapping", "build_mapping")

    return workflow.compile()


def build_lyrics_translation_graph(analyzer, llm, config):
    """
    Build the constraint-based lyrics translation graph - processes line by line with tools

    Args:
        analyzer: LyricsAnalyzer instance
        llm: LLM instance (ChatOllama)
        config: LyricsTranslationAgentConfig instance

    Returns:
        Compiled LangGraph
    """
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    # Create tool functions that LLM can call
    def count_syllables_tool(text: str, language: str) -> dict:
        """Count syllables in text"""
        count = analyzer.count_syllables(text, language)
        return {"text": text, "syllable_count": count}

    def verify_translation_tool(
        translation: str, target_syllables: int, language: str
    ) -> dict:
        """Verify if translation has target syllable count"""
        actual = analyzer.count_syllables(translation, language)
        passed = actual == target_syllables
        diff = actual - target_syllables
        if diff > 0:
            feedback = f"Too many! Remove {diff} syllable(s)."
        elif diff < 0:
            feedback = f"Too few! Add {abs(diff)} syllable(s)."
        else:
            feedback = "Perfect match!"
        return {
            "translation": translation,
            "target": target_syllables,
            "actual": actual,
            "passed": passed,
            "feedback": feedback,
        }

    # Define tools schema for LLM
    tools = [
        {
            "type": "function",
            "function": {
                "name": "count_syllables",
                "description": "Count syllables in the given text. Use this to check your translation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to count syllables in",
                        }
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verify_translation",
                "description": "Verify if translation has the correct syllable count. Returns pass/fail.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "translation": {
                            "type": "string",
                            "description": "Your translation to verify",
                        },
                        "target_syllables": {
                            "type": "integer",
                            "description": "The target syllable count",
                        },
                    },
                    "required": ["translation", "target_syllables"],
                },
            },
        },
    ]

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

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

        print(
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
                print(
                    f'      ✓ Attempt {attempt + 1}: "{translation}" ({actual}/{target_count})'
                )
                break
            else:
                print(
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
