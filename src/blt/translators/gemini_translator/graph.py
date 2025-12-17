"""Simple Gemini translation graph with validate-retry flow using LangChain"""

import time
import logging
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from .models import GeminiTranslationState
from .validator import Validator
from ..shared import (
    count_syllables,
    extract_rhyme_ending,
    detect_rhyme_scheme,
    get_syllable_patterns,
    LyricsAnalyzer,
)

logger = logging.getLogger(__name__)


def create_initial_state(
    source_lyrics: str,
    source_lang: str,
    target_lang: str,
    syllable_counts: list[int],
    rhyme_scheme: str,
    syllable_patterns: list[list[int]],
) -> GeminiTranslationState:
    """Create initial state for translation graph

    Args:
        source_lyrics: Source lyrics text
        source_lang: Source language code
        target_lang: Target language code
        syllable_counts: Target syllable counts
        rhyme_scheme: Target rhyme scheme
        syllable_patterns: Target syllable patterns

    Returns:
        GeminiTranslationState initialized
    """
    return {
        "source_lyrics": source_lyrics,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "syllable_counts": syllable_counts,
        "rhyme_scheme": rhyme_scheme,
        "syllable_patterns": syllable_patterns,
        "translated_lines": None,
        "reasoning": None,
        "translation_syllable_counts": None,
        "translation_rhyme_scheme": None,
        "validation_passed": None,
        "validation_details": None,
        "attempt": 1,
        "max_attempts": 3,
        "messages": [],
    }


def build_graph(llm, validator: Validator = None):
    """Build simple translation graph with validate-retry flow

    Binds 4 constraint validation tools to the LLM:
    - count_syllables
    - extract_rhyme_ending
    - detect_rhyme_scheme
    - get_syllable_patterns

    Graph structure:
    generate → validate → (check: passed?) → END
                             ↓ (no)
                          retry_check → (attempts < max?) → wait → generate
                             ↓ (no)
                             END

    Args:
        llm: LangChain ChatGoogleGenerativeAI (tools will be bound here)
        validator: Validator instance (creates new if None)

    Returns:
        Compiled LangGraph workflow
    """
    # Initialize validator
    validator = validator or Validator(LyricsAnalyzer())

    # Bind constraint tools to LLM
    constraint_tools = [
        count_syllables,
        extract_rhyme_ending,
        detect_rhyme_scheme,
        get_syllable_patterns,
    ]
    llm_with_tools = llm.bind_tools(constraint_tools)

    graph = StateGraph(GeminiTranslationState)

    def generate_node(state: GeminiTranslationState) -> dict:
        """Generate translation using Gemini LLM with tools"""
        print(
            f"   📝 Attempt {state['attempt']}/{state['max_attempts']}: Generating translation..."
        )
        logger.info(
            f"   📝 Attempt {state['attempt']}/{state['max_attempts']}: Generating translation..."
        )

        source_lines = [
            line.strip()
            for line in state["source_lyrics"].strip().split("\n")
            if line.strip()
        ]
        print(f"   📄 Source lines: {len(source_lines)}")
        target_lang = state["target_lang"]
        syllable_counts = state["syllable_counts"] or []
        rhyme_scheme = state["rhyme_scheme"] or ""
        syllable_patterns = state["syllable_patterns"] or []

        # Build prompt
        print(f"   🔧 Building prompt...")
        lines_with_targets = "\n".join(
            f'{i + 1}. "{line}" → {syllable_counts[i] if i < len(syllable_counts) else "?"} syllables'
            for i, line in enumerate(source_lines)
        )

        patterns_str = ""
        if syllable_patterns:
            patterns_str = "\n".join(
                f"  Line {i + 1}: {pattern}"
                for i, pattern in enumerate(syllable_patterns[: len(source_lines)])
            )

        prompt = f"""You are a professional lyrics translator. Translate ALL {len(source_lines)} lines to {target_lang}.

CRITICAL: Translate ALL {len(source_lines)} lines exactly. Do not skip any.

SOURCE LYRICS (with target syllable counts):
{lines_with_targets}

CONSTRAINT 1: SYLLABLE COUNTS (MUST MATCH EXACTLY)
Each line must have EXACTLY the specified syllable count.

CONSTRAINT 2: RHYME SCHEME
Rhyme pattern: {rhyme_scheme if rhyme_scheme else "preserve original"}

CONSTRAINT 3: SYLLABLE PATTERNS
{patterns_str if patterns_str else "Maintain natural distribution"}

Use the available tools to verify:
1. Syllable counts match exactly
2. Rhyme scheme is correct
3. Syllable patterns are maintained

OUTPUT FORMAT:
You MUST output exactly {len(source_lines)} translated lines, one per line.
Format each line as: [number]. [translation]
Example:
1. First line translation
2. Second line translation
...

IMPORTANT:
- Translate ALL {len(source_lines)} lines
- Do not stop after one line
- Each line must be numbered 1-{len(source_lines)}
- No extra explanations"""

        # Run agentic loop with tools
        print(f"   💬 Invoking LLM with tools...")
        messages = [
            SystemMessage(
                content="You are a lyrics translator. Use available tools to verify constraints."
            ),
            HumanMessage(content=prompt),
        ]

        print(f"   ⏳ Waiting for LLM response (this may take a moment)...")
        response = llm_with_tools.invoke(messages)
        print(f"   ✓ LLM response received")

        # Process tool calls in loop
        max_iterations = 10
        iteration = 0
        while response.tool_calls and iteration < max_iterations:
            iteration += 1
            print(
                f"   🔄 Tool call iteration {iteration}/{max_iterations}: {len(response.tool_calls)} tool(s) to execute"
            )

            # Add assistant response with tool calls
            messages.append(
                AIMessage(
                    content=response.content or "", tool_calls=response.tool_calls
                )
            )

            # Execute tools
            for tool_idx, tool_call in enumerate(response.tool_calls, 1):
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                print(f"      Tool {tool_idx}: {tool_name}")

                # Execute shared constraint tools
                if tool_name == "count_syllables":
                    result = count_syllables.invoke(tool_args)
                elif tool_name == "extract_rhyme_ending":
                    result = extract_rhyme_ending.invoke(tool_args)
                elif tool_name == "detect_rhyme_scheme":
                    result = detect_rhyme_scheme.invoke(tool_args)
                elif tool_name == "get_syllable_patterns":
                    result = get_syllable_patterns.invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                print(f"      → Result: {str(result)[:100]}...")
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

            # Invoke LLM again
            print(f"   ⏳ Invoking LLM again with tool results...")
            response = llm_with_tools.invoke(messages)
            print(f"   ✓ LLM response received")

        # Parse translations from response
        print(f"   📥 Parsing translations from response...")
        print(f"   📄 Raw LLM response:\n{response.content[:500]}")
        translated_lines = _extract_translations(response.content, len(source_lines))
        print(
            f"   ✓ Extracted {len(translated_lines)}/{len(source_lines)} translations"
        )
        if translated_lines:
            print(f"      Translations: {translated_lines[:3]}...")
        else:
            print(
                f"      ⚠️  NO TRANSLATIONS EXTRACTED - Response may be in wrong format"
            )

        logger.info(f"   ✓ Generated {len(translated_lines)} translations")

        return {
            "translated_lines": translated_lines,
            "reasoning": response.content or "",
            "messages": [f"Generated translation (attempt {state['attempt']})"],
        }

    def validate_node(state: GeminiTranslationState) -> dict:
        """Validate translation against all constraints using Validator"""
        print("   🔍 Validating translation...")
        logger.info("   ✓ Validating translation...")

        translated_lines = state["translated_lines"] or []
        target_lang = state["target_lang"]

        print(f"      Lines to validate: {len(translated_lines)}")
        print(f"      Syllables target: {state['syllable_counts']}")
        print(f"      Rhyme scheme target: {state['rhyme_scheme']}")

        # Use Validator to verify all constraints
        print(f"   ⏳ Running constraint verification...")
        validation_result = validator.verify_all_constraints(
            lines=translated_lines,
            language=target_lang,
            target_syllables=state["syllable_counts"],
            rhyme_scheme=state["rhyme_scheme"],
            target_patterns=state["syllable_patterns"],
        )

        all_passed = validation_result.get("passed", False)
        print(
            f"   {'✓' if all_passed else '⚠'} Validation complete "
            f"(score: {validation_result.get('overall_score', 0):.2f})"
        )
        logger.info(
            f"   {'✓' if all_passed else '⚠'} Validation complete "
            f"(score: {validation_result.get('overall_score', 0):.2f})"
        )

        return {
            "validation_passed": all_passed,
            "validation_details": validation_result,
            "messages": ["Validation complete"],
        }

    def should_retry(state: GeminiTranslationState) -> str:
        """Check if we should retry or end"""
        if state["validation_passed"]:
            print("   ✨ Translation validated successfully!")
            logger.info("   ✨ Translation validated successfully!")
            return "end"

        if state["attempt"] >= state["max_attempts"]:
            print(f"   ⚠ Max attempts ({state['max_attempts']}) reached")
            logger.warning(f"   ⚠ Max attempts ({state['max_attempts']}) reached")
            return "end"

        print(
            f"   ↻ Retrying... (attempt {state['attempt'] + 1}/{state['max_attempts']})"
        )
        logger.info(
            f"   ⏳ Retrying... (attempt {state['attempt'] + 1}/{state['max_attempts']})"
        )
        return "retry"

    def wait_node(state: GeminiTranslationState) -> dict:
        """Wait 3 seconds before retry"""
        print(f"   ⏱ Waiting 3 seconds before retry...")
        logger.info("   ⏱ Waiting 3 seconds before retry...")
        time.sleep(3)
        print(f"   ✓ Resuming translation...")
        return {
            "attempt": state["attempt"] + 1,
            "messages": ["Waited 3 seconds"],
        }

    # Add nodes
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("wait", wait_node)

    # Add edges
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        should_retry,
        {
            "end": END,
            "retry": "wait",
        },
    )
    graph.add_edge("wait", "generate")

    # Set entry point
    graph.set_entry_point("generate")

    return graph.compile()


def _extract_translations(response_text: str, expected_count: int) -> list[str]:
    """Extract numbered translations from LLM response

    Looks for patterns like:
    1. Translation line 1
    2. Translation line 2
    ...
    """
    import re

    lines = []
    print(f"   🔎 Searching for numbered translations in response...")

    for line_idx, line in enumerate(response_text.split("\n")):
        original_line = line
        line = line.strip()

        # Try to match numbered format: "N. translation"
        match = re.match(r"^\d+\.\s*(.+)$", line)
        if match:
            translation = match.group(1).strip()
            lines.append(translation)
            print(f"      Found line {len(lines)}: {translation[:50]}...")
            if len(lines) == expected_count:
                break

    if not lines:
        print(f"   ⚠️  Pattern not found in response")
        # Try alternative: look for any text in【Translation】 section
        if "【Translation】" in response_text:
            print(f"   🔍 Found 【Translation】 section, extracting...")
            start = response_text.find("【Translation】")
            end = response_text.find("【", start + 1)
            section = response_text[start:end] if end != -1 else response_text[start:]

            for line in section.split("\n")[1:]:  # Skip header
                line = line.strip()
                if line and not line.startswith("【"):
                    match = re.match(r"^\d+\.\s*(.+)$", line)
                    if match:
                        lines.append(match.group(1).strip())
                        print(
                            f"      Alternative extract line {len(lines)}: {lines[-1][:50]}..."
                        )
                        if len(lines) == expected_count:
                            break

    return lines
