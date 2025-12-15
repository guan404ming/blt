"""
Text-to-IPA Conversion using Phonemizer + espeak-ng

Installation:
    uv pip install phonemizer
    brew install espeak-ng

Language support: 100+ languages via espeak-ng
"""

import re
from phonemizer import phonemize
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


def count_syllables_from_ipa(ipa_text):
    """
    Count syllables in IPA text by counting vowel nuclei.

    Syllable核心是元音 (vowels). 我們計算IPA中的元音符號數量來估計音節數。

    IPA vowels include:
    - Monophthongs: i, y, ɨ, ʉ, ɯ, u, ɪ, ʏ, ʊ, e, ø, ɘ, ɵ, ɤ, o, ə, ɛ, œ, ɜ, ɞ, ʌ, ɔ, æ, ɐ, a, ɶ, ɑ, ɒ
    - Diphthongs are counted as single syllables (e.g., aɪ, eɪ, ɔɪ)
    """
    # IPA vowel pattern (包含所有常見的元音符號)
    vowel_pattern = r"[iɪeɛæaɑɒɔoʊuʉɨəɜɞʌyøœɶɐ]"

    # Find all vowels in the IPA text
    vowels = re.findall(vowel_pattern, ipa_text)

    return len(vowels)


# Examples with espeak-ng language codes
# Common language codes:
# - en-us: American English
# - en-gb: British English
# - de: German
# - fr-fr: French
# - es: Spanish
# - it: Italian
# - ja: Japanese
# - ko: Korean
# - cmn: Mandarin Chinese
# - pt: Portuguese
# - ru: Russian
# Full list: https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md

# 1. 英文範例 (English - American)
english_text = "Lyrics translation is a challenging task."
ipa_en = phonemize(english_text, language="en-us", backend="espeak", strip=True)
syllables_en = count_syllables_from_ipa(ipa_en)

print("--- 英文 (English - en-us) ---")
print(f"原文:   {english_text}")
print(f"IPA:    {ipa_en}")
print(f"音節數: {syllables_en}")
print("-" * 50)

# 2. 德文範例 (German)
german_text = "Ich möchte ein Bier trinken."
ipa_de = phonemize(german_text, language="de", backend="espeak", strip=True)
syllables_de = count_syllables_from_ipa(ipa_de)

print("--- 德文 (German - de) ---")
print(f"原文:   {german_text}")
print(f"IPA:    {ipa_de}")
print(f"音節數: {syllables_de}")
print("-" * 50)

# 3. 韓文範例 (Korean)
korean_text = "감사합니다"  # 謝謝
ipa_ko = phonemize(korean_text, language="ko", backend="espeak", strip=True)
syllables_ko = count_syllables_from_ipa(ipa_ko)

print("--- 韓文 (Korean - ko) ---")
print(f"原文:   {korean_text}")
print(f"IPA:    {ipa_ko}")
print(f"音節數: {syllables_ko}")
print("-" * 50)

# 4. 多個範例測試 (Additional examples)
print("\n--- 額外測試 (Additional Tests) ---")
test_cases = [
    ("Hello world", "en-us"),
    ("Bonjour", "fr-fr"),
    ("Hola", "es"),
    ("Beautiful", "en-us"),
    ("音樂翻譯", "cmn"),
]

for text, lang in test_cases:
    ipa = phonemize(text, language=lang, backend="espeak", strip=True)
    syllables = count_syllables_from_ipa(ipa)
    print(f"{text:15} ({lang:6}) -> IPA: {ipa:25} -> 音節: {syllables}")
print("-" * 50)
