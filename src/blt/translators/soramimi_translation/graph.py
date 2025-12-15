"""
Mapping-based soramimi (空耳) translation graph following ReAct pattern
"""

import json
import logging
from langgraph.graph import StateGraph, END
from .tools import create_soramimi_tools
from .utils import load_mapping
from .models import SoramimiTranslationState

logger = logging.getLogger(__name__)


def create_initial_state(
    source_lines: list[str],
    source_lang: str,
    target_lang: str,
    max_attempts: int,
    threshold: float,
) -> SoramimiTranslationState:
    """
    Create initial state for soramimi mapping graph

    Args:
        source_lines: List of source lyrics lines
        source_lang: Source language code
        target_lang: Target language code
        max_attempts: Maximum number of mapping refinement attempts
        threshold: Similarity threshold for stopping refinement

    Returns:
        SoramimiTranslationState initialized with all required fields
    """
    return {
        "source_lines": source_lines,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "source_phonemes": [],
        "phoneme_mapping": {},
        "mapping_scores": {},
        "soramimi_lines": None,
        "source_ipa": None,
        "target_ipa": None,
        "similarity_scores": None,
        "overall_similarity": None,
        "best_mapping": None,
        "best_lines": None,
        "best_scores": None,
        "best_ipas": None,
        "attempt": 1,
        "max_attempts": max_attempts,
        "threshold": threshold,
        "messages": [],
    }


def build_graph(analyzer, validator, llm):
    """
    Build the mapping-based soramimi translation graph using ReAct pattern.

    The graph uses a reasoning-acting cycle where the LLM reasons about phoneme
    mappings, uses tools to verify phonetic similarity, and iteratively refines
    the mapping until reaching the similarity threshold.

    Args:
        analyzer: LyricsAnalyzer instance
        validator: SoramimiValidator instance
        llm: LLM instance (ChatOllama)

    Returns:
        Compiled LangGraph workflow
    """
    # Create and bind tools
    tools = create_soramimi_tools(analyzer, validator)
    llm.bind_tools(tools)

    # Cache for fallback mappings per language
    fallback_cache = {}

    def get_fallback_mapping(target_lang: str) -> dict[str, str]:
        """Get or load fallback mapping for target language"""
        if target_lang not in fallback_cache:
            fallback_cache[target_lang] = load_mapping(target_lang)
        return fallback_cache[target_lang]

    def extract_phonemes_node(state: SoramimiTranslationState) -> dict:
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

    def build_mapping_node(state: SoramimiTranslationState) -> dict:
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

    def apply_mapping_node(state: SoramimiTranslationState) -> dict:
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

    def validate_node(state: SoramimiTranslationState) -> dict:
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

    def refine_mapping_node(state: SoramimiTranslationState) -> dict:
        """Increment attempt counter"""
        return {"attempt": state["attempt"] + 1}

    def should_continue(state: SoramimiTranslationState) -> str:
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

    def get_initial_mapping_prompt(state: SoramimiTranslationState) -> str:
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

    def get_refinement_prompt(state: SoramimiTranslationState) -> str:
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
    workflow = StateGraph(SoramimiTranslationState)

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
