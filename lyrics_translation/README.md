# Lyrics Translation with PydanticAI + Gemini 2.0 Flash

基於 LLM 的可唱歌詞翻譯系統 Baseline 實作

## 技術棧

- **PydanticAI**: Type-safe LLM 互動框架
- **Gemini 2.0 Flash**: Google 最新高效 LLM
- **LangChain**: RAG 和向量檢索
- **Pydantic**: 資料驗證和結構化輸出

## 安裝

```bash
# 建立虛擬環境
uv venv --python 3.11
source .venv/bin/activate

# 安裝依賴
uv pip install pydantic-ai-slim[gemini] langchain langchain-community langchain-google-genai faiss-cpu pypinyin syllables
```

## 快速開始

```python
from lyrics_translation import LyricsTranslator

# 初始化翻譯器
translator = LyricsTranslator(
    model="gemini-2.0-flash-exp",
    api_key="YOUR_GEMINI_API_KEY"
)

# 翻譯歌詞
result = translator.translate(
    source_lyrics="Let it go, let it go\nCan't hold it back anymore",
    source_lang="English",
    target_lang="Chinese"
)

print(result.translated_lines)
print(result.syllable_counts)
print(result.rhyme_endings)
```

## 功能特色

- ✅ Zero-shot 翻譯 (無需訓練)
- ✅ 自動音樂特徵提取 (音節數、押韻、停頓)
- ✅ Structured output (Pydantic models)
- ✅ 自動約束驗證與迭代
- ✅ RAG 範例檢索 (可選)
- ✅ 支援多語言到多語言

## 項目結構

```
lyrics_translation/
├── __init__.py
├── models.py           # Pydantic models
├── translator.py       # 核心翻譯器
├── feature_extractor.py # 音樂特徵提取
├── validator.py        # 約束驗證
└── rag.py             # RAG 範例檢索
```
