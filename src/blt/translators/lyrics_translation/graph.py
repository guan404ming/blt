"""
Constraint-based lyrics translation graph following ReAct pattern
"""

import logging
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from ..shared.tools import (
    count_syllables,
    text_to_ipa,
    extract_rhyme_ending,
    check_rhyme,
    get_syllable_patterns,
)
from .models import LyricsTranslationState

logger = logging.getLogger(__name__)


def create_initial_state(
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
        "current_refinement_idx": 0,
        "messages": [],
    }


def build_graph(analyzer, validator, llm, config):
    """
    Build the constraint-based lyrics translation graph using ReAct pattern.

    The graph uses a two-phase approach:
    1. Initial translation: LLM translates all lines at once considering global constraints
    2. Refinement: Line-by-line verification and correction to meet syllable/rhyme targets

    Args:
        analyzer: LyricsAnalyzer instance
        validator: Validator instance
        llm: LLM instance (ChatOllama)
        config: LyricsTranslationAgentConfig instance

    Returns:
        Compiled LangGraph workflow
    """
    # Bind essential tools to LLM for verification and analysis
    tools = [
        count_syllables,
        text_to_ipa,
        extract_rhyme_ending,
        check_rhyme,
        get_syllable_patterns,
    ]
    llm_with_tools = llm.bind_tools(tools)

    def initial_translation_node(state: LyricsTranslationState) -> dict:
        """Translate all lines at once considering 3 constraints: syllables, rhyme, patterns"""
        source_lines = [
            line.strip()
            for line in state["source_lyrics"].strip().split("\n")
            if line.strip()
        ]
        target_lang = state["target_lang"]
        syllable_counts = state["syllable_counts"] or []
        rhyme_scheme = state["rhyme_scheme"] or ""
        syllable_patterns = state["syllable_patterns"] or []

        logger.info("   🌍 Phase 1: Initial translation (considering 3 constraints)...")

        # Build comprehensive prompt with all 3 constraints
        lines_with_targets = "\n".join(
            f'{i + 1}. "{line}" → {syllable_counts[i] if i < len(syllable_counts) else "?"} syllables'
            for i, line in enumerate(source_lines)
        )

        # Format syllable patterns
        patterns_str = ""
        if syllable_patterns:
            patterns_str = "\n".join(
                f"  Line {i + 1}: {pattern}"
                for i, pattern in enumerate(syllable_patterns[: len(source_lines)])
            )

        prompt = f"""You are a professional lyrics translator. Translate ALL the following lyrics to {target_lang} while meeting ALL 3 musical constraints.

SOURCE LYRICS:
{lines_with_targets}

CONSTRAINT 1: SYLLABLE COUNTS (MUST MATCH EXACTLY)
Each line must have EXACTLY the specified syllable count above.

CONSTRAINT 2: RHYME SCHEME
Rhyme pattern: {rhyme_scheme if rhyme_scheme else "preserve the original rhyme scheme"}

CONSTRAINT 3: SYLLABLE PATTERNS (words with specified syllable distribution)
{patterns_str if patterns_str else "Maintain natural word syllable distribution"}

QUALITY REQUIREMENTS:
- Maintain poetic quality and emotional impact
- Preserve musical flow and rhythm
- Ensure natural language (not forced/artificial)

Use the available tools to verify:
1. Syllable counts match exactly
2. Rhyme scheme is correct
3. Syllable patterns are maintained

OUTPUT FORMAT:
Return ONLY the translated lines, one per line, WITHOUT numbered prefixes.
Do not include "1.", "2.", etc. - just the translation text.
Do not include any explanations or notes."""

        # Run agentic loop: invoke LLM, process tool calls, repeat until we get content
        messages = [
            SystemMessage(content=config.get_system_prompt()),
            HumanMessage(content=prompt),
        ]

        response = llm_with_tools.invoke(messages)

        # Process tool calls in a loop until LLM provides actual translations
        max_iterations = 10
        iteration = 0
        while response.tool_calls and iteration < max_iterations:
            iteration += 1

            # Add assistant response with tool calls to messages
            messages.append(
                AIMessage(
                    content=response.content or "", tool_calls=response.tool_calls
                )
            )

            # Build and add tool results
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                # Execute the tool
                if tool_name == "count_syllables":
                    result = count_syllables.invoke(tool_args)
                elif tool_name == "text_to_ipa":
                    result = text_to_ipa.invoke(tool_args)
                elif tool_name == "extract_rhyme_ending":
                    result = extract_rhyme_ending.invoke(tool_args)
                elif tool_name == "check_rhyme":
                    result = check_rhyme.invoke(tool_args)
                elif tool_name == "get_syllable_patterns":
                    result = get_syllable_patterns.invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

            # Invoke LLM again with tool results
            response = llm_with_tools.invoke(messages)

        # Parse translations from response
        translated_lines = _extract_translations(response.content, len(source_lines))

        logger.info(
            f"   ✓ Generated {len(translated_lines)} initial translations (3 constraints)"
        )

        return {
            "translated_lines": translated_lines,
            "current_refinement_idx": 0,
        }

    def refine_line_node(state: LyricsTranslationState) -> dict:
        """Refine one line at a time - focus ONLY on syllable count"""
        source_lines = [
            line.strip()
            for line in state["source_lyrics"].strip().split("\n")
            if line.strip()
        ]
        target_syllables = state["syllable_counts"] or []
        translated_lines = list(state.get("translated_lines") or [])
        current_idx = state.get("current_refinement_idx", 0)
        target_lang = state["target_lang"]

        if current_idx >= len(source_lines):
            return {"translated_lines": translated_lines}

        # Ensure translated_lines has enough slots for all source lines
        while len(translated_lines) < len(source_lines):
            translated_lines.append("")

        source_line = source_lines[current_idx]
        target_count = (
            target_syllables[current_idx] if current_idx < len(target_syllables) else 0
        )
        current_translation = (
            translated_lines[current_idx] if current_idx < len(translated_lines) else ""
        )

        logger.info(
            f"   🔧 Refining syllables: Line {current_idx + 1}/{len(source_lines)}: {target_count} syllables"
        )

        # Check initial translation
        actual = analyzer.count_syllables(current_translation, target_lang)

        if actual == target_count:
            logger.info(
                f'      ✓ Line {current_idx + 1} already correct: "{current_translation}" ({actual}/{target_count})'
            )
            return {
                "translated_lines": translated_lines,
                "current_refinement_idx": current_idx + 1,
            }

        # Iterative refinement - focus only on syllable count
        max_attempts = 10
        best_translation = current_translation
        best_diff = abs(actual - target_count)

        for attempt in range(max_attempts):
            # Calculate feedback for syllable adjustment
            actual = analyzer.count_syllables(best_translation, target_lang)
            diff = actual - target_count

            if diff > 0:
                feedback = f"Too long by {diff} syllable(s). Remove words or use shorter alternatives."
            else:
                feedback = f"Too short by {abs(diff)} syllable(s). Add words or use longer alternatives."

            prompt = f"""Adjust this translation to have EXACTLY {target_count} syllables.
Focus ONLY on syllable count - minimize meaning changes.

Original: "{source_line}"
Current: "{best_translation}"
Current syllables: {actual}/{target_count}
Action: {feedback}

STRATEGIES:
- If too long: remove adjectives, use shorter words, merge concepts
- If too short: add descriptive words, use longer characters, expand descriptions

Keep the core meaning as close as possible to the current translation.
Output ONLY the adjusted translation (no quotes, no explanations)."""

            # Run agentic loop for refinement
            messages = [
                SystemMessage(content=config.get_system_prompt()),
                HumanMessage(content=prompt),
            ]
            response = llm_with_tools.invoke(messages)

            # Process tool calls in agentic loop
            max_refinement_iterations = 5
            refinement_iteration = 0
            while (
                response.tool_calls and refinement_iteration < max_refinement_iterations
            ):
                refinement_iteration += 1
                messages.append(
                    AIMessage(
                        content=response.content or "", tool_calls=response.tool_calls
                    )
                )

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    if tool_name == "count_syllables":
                        result = count_syllables.invoke(tool_args)
                    elif tool_name == "text_to_ipa":
                        result = text_to_ipa.invoke(tool_args)
                    elif tool_name == "extract_rhyme_ending":
                        result = extract_rhyme_ending.invoke(tool_args)
                    elif tool_name == "check_rhyme":
                        result = check_rhyme.invoke(tool_args)
                    elif tool_name == "get_syllable_patterns":
                        result = get_syllable_patterns.invoke(tool_args)
                    else:
                        result = f"Unknown tool: {tool_name}"

                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_id)
                    )

                response = llm_with_tools.invoke(messages)

            # Extract translation
            import re

            translation = response.content.strip()

            # Remove quotes
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1]

            # Remove code blocks
            if translation.startswith("```") or translation.startswith("{"):
                match = re.search(r'"([^"]+)"', translation)
                if match:
                    translation = match.group(1)
                else:
                    for line in translation.split("\n"):
                        line = line.strip()
                        if (
                            line
                            and not line.startswith("{")
                            and not line.startswith("```")
                        ):
                            translation = line
                            break

            # Remove numbered prefix (e.g., "1. text" -> "text")
            translation = re.sub(r"^\d+\.\s*", "", translation).strip()

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

        translated_lines[current_idx] = best_translation
        return {
            "translated_lines": translated_lines,
            "current_refinement_idx": current_idx + 1,
        }

    def check_refinement_progress_node(state: LyricsTranslationState) -> dict:
        """Check if all lines have been refined"""
        source_lines = [
            line.strip()
            for line in state["source_lyrics"].strip().split("\n")
            if line.strip()
        ]
        current_idx = state.get("current_refinement_idx", 0)
        return {"all_lines_done": current_idx >= len(source_lines)}

    def calculate_metrics_node(state: LyricsTranslationState) -> dict:
        """Calculate final translation metrics"""
        if not state.get("translated_lines"):
            return {
                "translation_syllable_counts": [],
                "translation_rhyme_endings": [],
                "translation_syllable_patterns": [],
                "reasoning": "No lines were translated.",
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

        # Build reasoning summary
        reasoning = f"Translated {len(translated_lines)} lines with {matches}/{len(expected)} matching target syllable counts ({score:.0%} match rate)."

        return {
            "translation_syllable_counts": syllable_counts,
            "translation_rhyme_endings": rhyme_endings,
            "translation_syllable_patterns": syllable_patterns,
            "reasoning": reasoning,
        }

    def should_continue_refinement(state: LyricsTranslationState) -> str:
        """Decide whether to refine next line or finish"""
        if state.get("all_lines_done"):
            return "calculate"
        return "refine"

    # Build workflow
    workflow = StateGraph(LyricsTranslationState)

    # Add nodes for two-phase approach
    workflow.add_node("initial_translation", initial_translation_node)
    workflow.add_node("refine_line", refine_line_node)
    workflow.add_node("check_refinement", check_refinement_progress_node)
    workflow.add_node("calculate_metrics", calculate_metrics_node)

    # Set entry point
    workflow.set_entry_point("initial_translation")

    # Define edges for two-phase flow
    workflow.add_edge("initial_translation", "refine_line")
    workflow.add_edge("refine_line", "check_refinement")
    workflow.add_conditional_edges(
        "check_refinement",
        should_continue_refinement,
        {"refine": "refine_line", "calculate": "calculate_metrics"},
    )
    workflow.add_edge("calculate_metrics", END)

    compiled = workflow.compile()
    mermaid_str = compiled.get_graph().draw_mermaid()
    print(mermaid_str)

    return compiled


def _extract_translations(text: str, num_lines: int) -> list[str]:
    """
    Extract translated lines from LLM response.

    Tries to parse numbered list or newline-separated lines.
    Removes numbered prefixes like "1.", "2.", etc.
    """
    import re

    lines = []

    # Try to match numbered format: 1. "translation" or 1. translation
    # Captures text after the number and dot, removing the number
    pattern = r'^\s*\d+\.\s*["\']?([^"\']*?)["\']?\s*$'
    for match in re.finditer(pattern, text, re.MULTILINE):
        extracted = match.group(1).strip()
        if extracted:
            lines.append(extracted)

    # If not enough lines, try simple line splitting and remove any leading numbers
    if len(lines) < num_lines:
        raw_lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in raw_lines:
            # Remove leading number and dot pattern: "1. text" -> "text"
            cleaned = re.sub(r"^\d+\.\s*", "", line).strip()
            if cleaned:
                lines.append(cleaned)

    # Remove duplicates while preserving order
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    # Trim to exact number of lines needed
    return unique_lines[:num_lines]
