"""Configuration for Gemini Lyrics Translator"""

from dataclasses import dataclass


@dataclass
class GeminiTranslatorConfig:
    """Configuration for Gemini Lyrics Translator"""

    # API settings
    api_key: str = ""  # Set via GEMINI_API_KEY environment variable
    model: str = "gemini-2.5-flash"

    # Language defaults
    default_source_lang: str = "en-us"
    default_target_lang: str = "zh-tw"

    # Output settings
    auto_save: bool = False
    save_dir: str = "outputs"
    save_format: str = "json"

    # Translation settings
    max_retries: int = 3
    temperature: float = 0.7

    # ==================== PROMPT TEMPLATE ====================

    def get_prompt_template(self) -> str:
        """Generate prompt template for lyrics translation

        The template includes:
        - Translation requirements
        - Syllable count constraints
        - Rhyme scheme constraints
        - Syllable pattern constraints
        """
        return """You are a professional lyrics translator.

【Translation Task】

Translate the following lyrics from {source_lang} to {target_lang}.

【Source Lyrics】

{source_lyrics}

【Requirements】

1. Each line MUST match the exact syllable count:
{syllables_requirement}

2. Rhyme scheme: {rhyme_scheme}

3. Syllable patterns per line:
{patterns_requirement}

4. Format output as:
【Translation】

1. [translation line 1]
2. [translation line 2]
...

【Syllables】
Actual: [count per line]

【Rhymes】
Actual: [rhyme scheme]

【Patterns】
1. Actual: [pattern]
2. Actual: [pattern]
...

5. Important notes:
   - For Chinese: 1 character = 1 syllable
   - No punctuation in translations
   - Preserve meaning alignment
   - Strictly follow syllable counts
"""

    def get_validation_prompt(self, translation: str, constraints: dict) -> str:
        """Generate prompt for validation

        Args:
            translation: The translated text
            constraints: Dictionary with syllable_counts, rhyme_scheme, syllable_patterns

        Returns:
            Validation prompt
        """
        return f"""Validate this translation against the constraints.

【Translation】
{translation}

【Target Syllable Counts】
{constraints.get("syllable_counts", [])}

【Target Rhyme Scheme】
{constraints.get("rhyme_scheme", "N/A")}

【Target Syllable Patterns】
{constraints.get("syllable_patterns", [])}

【Validation Tasks】

1. Count syllables in each line
2. Check rhyme scheme
3. Verify syllable patterns
4. Rate overall quality (1-10)

Provide structured validation output with PASS/FAIL for each constraint.
"""
