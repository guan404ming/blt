"""
Text-to-IPA Conversion using Phonemizer + espeak-ng

Installation:
    uv pip install phonemizer
    brew install espeak-ng

Language support: 100+ languages via espeak-ng
"""

import os
from phonemizer import phonemize

# Set environment variables for phonemizer to find espeak-ng
os.environ["PHONEMIZER_ESPEAK_PATH"] = "/opt/homebrew/bin/espeak-ng"
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/lib/libespeak-ng.dylib"

# 1. 英文範例 (English - American)
english_text = "Lyrics translation is a challenging task."
ipa_en = phonemize(english_text, language="en-us", backend="espeak", strip=True)

print("--- 英文 (English - en-us) ---")
print(f"原文: {english_text}")
print(f"IPA:  {ipa_en}")
print("-" * 30)

# 2. 德文範例 (German)
german_text = "Ich möchte ein Bier trinken."
ipa_de = phonemize(german_text, language="de", backend="espeak", strip=True)

print("--- 德文 (German - de) ---")
print(f"原文: {german_text}")
print(f"IPA:  {ipa_de}")
print("-" * 30)

# 3. 韓文範例 (Korean)
korean_text = "감사합니다"  # 謝謝
ipa_ko = phonemize(korean_text, language="ko", backend="espeak", strip=True)

print("--- 韓文 (Korean - ko) ---")
print(f"原文: {korean_text}")
print(f"IPA:  {ipa_ko}")
print("-" * 30)
