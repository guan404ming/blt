from phonemizer import phonemize
from phonemizer.separator import Separator

# 1. 英文範例 (English Example)
english_text = "Love is a battlefield. We need to sing the song."

# 參數解釋：
# language='en': 指定輸入語言為英文。
# backend='espeak': 使用支援最多語言的 eSpeak-ng 引擎。
# strip=True: 移除輸出字符串兩端的空白。
# separator: 定義音節和單詞之間的間隔。
ipa_en = phonemize(
    english_text,
    language="en",
    backend="espeak",
    strip=True,
    separator=Separator(phone=" ", word=" | ", syllable="-"),
)

print("--- 英文 (English) ---")
print(f"原文: {english_text}")
print(f"IPA:  {ipa_en}")
print("-" * 20)

# 2. 中文範例 (Mandarin Example)
# 註：中文的 G2P 轉換可能需要額外的配置或專門的庫以獲得更高的準確度
# 但 eSpeak-ng 仍可以提供基礎轉換。
chinese_text = "今天天氣很好，我們去公園玩吧。"

ipa_zh = phonemize(
    chinese_text,
    language="zh",
    backend="espeak",
    strip=True,
    separator=Separator(phone=" ", word=" | ", syllable="-"),
)

print("--- 中文 (Mandarin) ---")
print(f"原文: {chinese_text}")
print(f"IPA:  {ipa_zh}")
print("-" * 20)

# 3. 日文範例 (Japanese Example)
japanese_text = "おはようございます"

ipa_ja = phonemize(
    japanese_text,
    language="ja",
    backend="espeak",
    strip=True,
    separator=Separator(phone=" ", word=" | ", syllable="-"),
)

print("--- 日文 (Japanese) ---")
print(f"原文: {japanese_text}")
print(f"IPA:  {ipa_ja}")
print("-" * 20)
