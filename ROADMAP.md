
## 歌詞翻譯功能規劃 (Lyrics Translation Plan)

### 目標
為音樂生成系統增加**可唱的 (Singable)** 多語言歌詞翻譯能力,支援任意語言對之間的翻譯,並確保翻譯結果能夠與音樂完美結合,使用戶能夠快速驗證翻譯品質並整合至音樂生成流程中。

### 研究導向設計原則 (基於 ACL 2023 + LLM Zero-shot)
1. **零樣本/少樣本學習**: 使用大型 LLM (無需 fine-tuning),僅透過 prompt engineering
2. **全自動化流程**: 系統自動提取音樂特徵並生成約束,無需人工標註
3. **模組化架構**: 可獨立測試與評估翻譯品質
4. **可擴展性**: 支援多種 LLM 與評估指標的比較
5. **可唱性優先 (Singability First)**: 遵循 "Pentathlon Principle" - 平衡可唱性、韻律、押韻、自然度、語義五大要素
6. **快速驗證**: 無需訓練,可立即部署測試

### 技術方案 (LLM Zero-shot 方法 + 自動化流程)

#### 階段一:LLM 模型選擇與 Prompt Engineering
**目標**: 使用現成 LLM 建立可唱歌詞翻譯能力 (無需 fine-tuning)

- **候選 LLM 模型**:
  - **GPT-4 / GPT-4 Turbo** [推薦]: 最強多語言能力,支援 function calling
  - **Claude 3 Opus/Sonnet**: 長上下文,指令遵循能力強
  - **Gemini 1.5 Pro**: 支援超長上下文 (1M tokens),適合複雜 prompt
  - **開源替代**:
    - **Qwen2.5-72B-Instruct**: 強大的多語言能力
    - **LLaMA 3.1-70B-Instruct**: 適合自部署
    - **DeepSeek-V2**: 中文能力優秀

- **Structured Output 設計** (確保輸出格式一致):

  **方法 1: JSON Mode (GPT-4, Gemini)**
  ```python
  from openai import OpenAI
  from pydantic import BaseModel, Field

  # 定義結構化輸出 schema
  class LyricTranslation(BaseModel):
      translated_lines: list[str] = Field(
          description="逐行翻譯結果"
      )
      syllable_counts: list[int] = Field(
          description="每行的實際音節數"
      )
      rhyme_endings: list[str] = Field(
          description="每行的韻腳"
      )
      reasoning: str = Field(
          description="翻譯思路和考量"
      )
      constraint_satisfaction: dict = Field(
          description="約束滿足情況 {length: bool, rhyme: bool, boundary: bool}"
      )

  # 呼叫 LLM
  response = client.chat.completions.create(
      model="gpt-4-turbo",
      messages=[
          {"role": "system", "content": "你是專業的歌詞翻譯專家"},
          {"role": "user", "content": prompt}
      ],
      response_format={"type": "json_object"},
      # 或使用 function calling
      tools=[{
          "type": "function",
          "function": {
              "name": "translate_lyrics",
              "parameters": LyricTranslation.model_json_schema()
          }
      }]
  )
  ```

  **方法 2: XML/Markdown Structured Output**
  ```markdown
  請按照以下格式輸出翻譯結果:

  <translation>
    <line number="1">
      <original>{line_1}</original>
      <translated>{translated_line_1}</translated>
      <syllables>{count}</syllables>
      <rhyme>{ending}</rhyme>
    </line>
    <line number="2">
      ...
    </line>
    <reasoning>
      {your_reasoning}
    </reasoning>
    <constraints>
      <length_satisfied>true/false</length_satisfied>
      <rhyme_satisfied>true/false</rhyme_satisfied>
      <boundary_satisfied>true/false</boundary_satisfied>
    </constraints>
  </translation>
  ```

  **方法 3: Instructor Library (推薦)**
  ```python
  import instructor
  from openai import OpenAI

  # Patch OpenAI client
  client = instructor.from_openai(OpenAI())

  # 自動解析為 Pydantic model
  translation = client.chat.completions.create(
      model="gpt-4-turbo",
      response_model=LyricTranslation,
      messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
      ]
  )

  # 直接得到 typed object
  print(translation.translated_lines)
  print(translation.syllable_counts)
  ```

- **Prompt 設計策略** (結合 structured output):
  ```markdown
  你是專業的歌詞翻譯專家。請將以下歌詞從 {source_lang} 翻譯為 {target_lang}，
  並以 JSON 格式輸出結果。

  【原始歌詞】
  {source_lyrics}

  【音樂約束】
  - 目標音節數: {syllable_count} (每行必須精確匹配)
  - 押韻方案: {rhyme_scheme} (例如: AABB, ABAB)
  - 音樂停頓位置: {pause_positions} (詞語邊界必須對齊)

  【翻譯要求】
  1. 保持原意和情感
  2. 符合目標語言的自然表達
  3. 嚴格遵守音節數限制
  4. 在指定位置押韻
  5. 避免在音樂停頓處斷詞

  請以 JSON 格式輸出，包含:
  - translated_lines: 逐行翻譯
  - syllable_counts: 每行音節數
  - rhyme_endings: 每行韻腳
  - reasoning: 翻譯思路
  - constraint_satisfaction: 約束滿足情況
  ```

- **Few-shot Learning** (可選):
  - 在 prompt 中加入 2-3 個高品質翻譯範例
  - 展示如何處理音節數、押韻、詞邊界的範例
  - 範例也使用相同的結構化格式

#### 階段二:全自動音樂特徵提取
**目標**: 從音樂自動提取約束條件,無需人工標註

- **音節數自動計算**:
  - **來源歌詞**: 使用音節分割工具
    - 英文: `pyphen`, `syllables` library
    - 中文: 字符數即音節數
    - 日文/韓文: 假名/音節分析器
  - **目標設定**: 保持源語言音節數或根據旋律音符數設定

- **押韻方案自動檢測**:
  - **源歌詞分析**:
    - 提取每行末字/詞的韻母
    - 使用 `pronouncing` (英文), `pypinyin` (中文) 等工具
    - 自動識別押韻模式 (AABB, ABAB, etc.)
  - **目標語言映射**:
    - 根據源語言押韻模式生成目標語言韻腳要求
    - 考慮目標語言的韻腳體系

- **音樂停頓位置自動標註**:
  - **從 MIDI/MusicXML 提取**:
    - 檢測 rest (休止符) 位置
    - 檢測下拍 (downbeat) 位置
    - 檢測長音符 (> 0.5 拍) 起始位置
  - **詞邊界建議**:
    - 在停頓位置前後建議詞邊界
    - 生成 "建議詞邊界位置" 列表給 LLM

- **自動化流程**:
  ```python
  def extract_music_constraints(source_lyrics, music_file=None):
      # 1. 音節數
      syllable_counts = count_syllables_per_line(source_lyrics)

      # 2. 押韻方案
      rhyme_scheme = detect_rhyme_scheme(source_lyrics)

      # 3. 音樂停頓 (如果有音樂檔案)
      if music_file:
          pause_positions = extract_pauses_from_music(music_file)
      else:
          # 基於標點符號和句子結構推測
          pause_positions = infer_pauses(source_lyrics)

      return {
          'syllable_counts': syllable_counts,
          'rhyme_scheme': rhyme_scheme,
          'pause_positions': pause_positions
      }
  ```

#### 階段三:品質評估框架
**目標**: 建立可重複的翻譯品質驗證流程 (遵循 Pentathlon Principle)

- **自動評估指標**:
  - **Sacre-BLEU**: 翻譯品質標準指標
  - **TER (Translation Edit Rate)**: 衡量後製編輯成本
  - **Length Accuracy (LA)**: 音節數準確度
  - **Rhyme Accuracy (RA)**: 押韻準確度
  - **Boundary Recall (BR)**: 詞邊界召回率

- **人工評估設計** (五點量表):
  1. **Sense (語義保真度)**: 翻譯是否保留原意
  2. **Naturalness (自然度)**: 是否像目標語言原創歌詞
  3. **Music-Lyric Compatibility (音樂兼容性)**: 歌詞與旋律的匹配度
  4. **Singable Translation Score (STS)**: 整體可唱性評分

- **評估流程**:
  - 句級評估: Sense, Naturalness, Compatibility
  - 段落級評估: STS (結合樂譜與合成演唱音檔)
  - Inter-rater agreement: ICC > 0.78 (good reliability)

#### 階段四:LLM 翻譯流程整合
**目標**: 建立端到端自動化翻譯流程

- **完整流程架構**:
  ```
  ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
  │ Source Lyrics   │ ──► │ Feature Extractor│ ──► │  LLM Translator │ ──► │  Validation  │
  │ + Music (opt.)  │     │  (自動提取約束)   │     │  (Zero-shot)    │     │  & Iteration │
  └─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
                                   │                          │                      │
                          ┌────────┴────────┐        ┌───────┴────────┐    ┌────────┴─────────┐
                          │ - 音節數        │        │ - Prompt 組裝   │    │ - 約束驗證       │
                          │ - 押韻方案      │        │ - API 呼叫      │    │ - 品質評分       │
                          │ - 停頓位置      │        │ - 結果解析      │    │ - 自動重試       │
                          └─────────────────┘        └─────────────────┘    └──────────────────┘
  ```

- **實作細節**:
  1. **輸入處理**:
     - 接收源歌詞文本
     - (可選) 接收 MIDI/MusicXML 音樂檔案
     - 自動語言檢測

  2. **約束提取** (全自動):
     ```python
     constraints = extract_music_constraints(
         source_lyrics=lyrics,
         music_file=midi_file  # 可選
     )
     ```

  3. **Prompt 生成**:
     ```python
     prompt = build_translation_prompt(
         source_lyrics=lyrics,
         source_lang=source_lang,
         target_lang=target_lang,
         constraints=constraints,
         few_shot_examples=examples  # 可選
     )
     ```

  4. **LLM 呼叫** (使用 structured output):
     ```python
     import instructor
     from openai import OpenAI
     from pydantic import BaseModel, Field

     # 定義輸出結構
     class LyricTranslation(BaseModel):
         translated_lines: list[str]
         syllable_counts: list[int]
         rhyme_endings: list[str]
         reasoning: str
         constraint_satisfaction: dict[str, bool]

     # Patch client
     client = instructor.from_openai(OpenAI())

     # 呼叫 LLM (自動解析為結構化輸出)
     translation = client.chat.completions.create(
         model="gpt-4-turbo",
         response_model=LyricTranslation,
         messages=[
             {"role": "system", "content": "你是專業的歌詞翻譯專家"},
             {"role": "user", "content": prompt}
         ],
         temperature=0.7,
         max_tokens=2000
     )

     # 直接訪問結構化資料
     print(translation.translated_lines)
     print(translation.syllable_counts)
     ```

  5. **結果驗證與迭代** (基於結構化輸出):
     ```python
     # 自動驗證約束滿足
     def validate_constraints(translation: LyricTranslation,
                             target_constraints: dict) -> dict:
         errors = []

         # 驗證音節數
         if translation.syllable_counts != target_constraints['syllable_counts']:
             errors.append({
                 'type': 'length',
                 'expected': target_constraints['syllable_counts'],
                 'actual': translation.syllable_counts
             })

         # 驗證押韻
         expected_rhyme = target_constraints['rhyme_scheme']
         if not check_rhyme_pattern(translation.rhyme_endings, expected_rhyme):
             errors.append({
                 'type': 'rhyme',
                 'expected': expected_rhyme,
                 'actual': translation.rhyme_endings
             })

         return {
             'passed': len(errors) == 0,
             'errors': errors
         }

     validation_result = validate_constraints(translation, constraints)

     if not validation_result['passed']:
         # 自動生成修正 prompt
         feedback_prompt = f"""
         你的翻譯有以下問題需要修正:
         {format_errors(validation_result['errors'])}

         原翻譯:
         {translation.translated_lines}

         請重新翻譯並確保滿足所有約束。
         """

         # 重試
         translation = client.chat.completions.create(
             model="gpt-4-turbo",
             response_model=LyricTranslation,
             messages=[
                 {"role": "system", "content": "你是專業的歌詞翻譯專家"},
                 {"role": "user", "content": feedback_prompt}
             ]
         )
     ```

- **整合至 MusicGen Pipeline**:
  ```
  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
  │   Prompt    │ ──► │ LLM Translate│ ──► │   Retrieval  │ ──► │   MusicGen  │ ──► │  Output  │
  │ (任何語言)   │     │  (→ English) │     │    (CLAP)    │     │    (ICL)    │     │          │
  └─────────────┘     └──────────────┘     └──────────────┘     └─────────────┘     └──────────┘
  ```

### 資料集需求 (用於評估與 Few-shot Examples)
- **無需訓練資料**: 使用 LLM zero-shot/few-shot,不需要 fine-tuning
- **推薦資料集** (用於評估與少樣本學習):
  1. **[facebook/flores](https://huggingface.co/datasets/facebook/flores)** [主要評估資料集]
     - 101 種語言的高品質平行翻譯語料 (3001 句專業翻譯)
     - **支援任意語言對組合**: 例如中文↔日文、韓文↔西班牙文等
     - 用途: 多語言翻譯品質評估的黃金標準,提供 BLEU/COMET baseline
     - 載入方式:
       ```python
       # 中文 → 英文
       load_dataset("facebook/flores", "zho_Hans-eng_Latn")
       # 日文 → 韓文
       load_dataset("facebook/flores", "jpn_Jpan-kor_Hang")
       # 任意語言對
       ```

  2. **[brunokreiner/genius-lyrics](https://huggingface.co/datasets/brunokreiner/genius-lyrics)**
     - 48萬首英文歌詞及 metadata
     - 用途: 歌詞領域測試資料,可翻譯為任意目標語言評估
     - 載入方式: `load_dataset("brunokreiner/genius-lyrics")`

  3. **[OpenLLM-France/Translation-Instruct](https://huggingface.co/datasets/OpenLLM-France/Translation-Instruct)**
     - 包含專業人工翻譯的歌詞片段 (法文-英文雙向)
     - 用途: 歌詞特定領域的翻譯品質 ground truth
     - 載入方式: `load_dataset("OpenLLM-France/Translation-Instruct")`

- **評估資料**:
  - 使用上述資料集建立測試集
  - 從 MusicCaps 抽取 50-100 筆音樂描述作為額外評估
  - 結合人工評估 20-30 筆翻譯樣本

### 驗證計畫
1. **翻譯品質驗證**:
   - 使用 BLEU/COMET 分數快速評估
   - 人工檢查 20-30 筆翻譯樣本

2. **端到端測試**:
   - 使用多語言 prompt 生成音樂
   - 比較翻譯前後的 CLAP score 差異
   - 評估翻譯是否保留音樂風格描述的關鍵資訊

3. **效能評估**:
   - 測量翻譯增加的延遲時間
   - 評估不同模型大小的權衡

### 預期成果 (LLM Zero-shot 方法)
- **多語言支援**: 支援至少 20+ 種主要語言的任意互譯
  - 亞洲語言: 中文 (簡/繁)、日文、韓文、泰文、越南文、印尼文
  - 歐洲語言: 英文、法文、西班牙文、德文、義大利文、俄文
  - 其他: 阿拉伯文、印地文、葡萄牙文等

- **約束控制準確度** (預期目標):
  - **Length Accuracy: 85-95%** (取決於 LLM 能力)
  - **Rhyme Accuracy: 80-90%** (可透過迭代提升)
  - **Boundary Recall: 70-85%** (透過 prompt 引導)
  - 註: 相較 fine-tuned 模型略低,但無需訓練

- **翻譯品質**:
  - **BLEU: 預期 25-35** (取決於語言對與 LLM 選擇)
  - **自然度**: LLM 通常優於專用翻譯模型
  - **創意性**: LLM 可產生更有詩意的表達

- **系統優勢**:
  - ✅ **零訓練成本**: 無需 GPU 訓練,立即可用
  - ✅ **全自動化**: 無需人工標註約束條件
  - ✅ **快速迭代**: 透過 prompt engineering 快速調整
  - ✅ **多語言擴展**: 支援 LLM 涵蓋的所有語言
  - ✅ **可解釋性**: LLM 可輸出推理過程
  - ⚠️ **API 成本**: 需考慮 LLM API 呼叫費用
  - ⚠️ **約束準確度**: 略低於 fine-tuned 模型

- **成本估算** (以 GPT-4 為例):
  - 每首歌 (~200 tokens input + ~300 tokens output): ~$0.02
  - 100 首歌翻譯: ~$2
  - 1000 首歌翻譯: ~$20
  - 使用開源 LLM (Qwen/LLaMA) 則無 API 成本

- **使用情境**:
  - 情境 1: 中文歌詞 + MIDI → 英文可唱翻譯
  - 情境 2: 日文歌詞 (無音樂) → 韓文翻唱版本 (自動推測約束)
  - 情境 3: 英文 → 10 種語言本地化 (批次翻譯)

### 風險與挑戰 (LLM 方法特有)
- **約束遵守不穩定**: LLM 可能不完全遵守音節數/押韻要求
  - 緩解方案: 使用驗證+迭代機制,自動重試
- **API 依賴與成本**: 商用 LLM 需要 API 費用
  - 緩解方案: 使用開源 LLM (Qwen, LLaMA) 自部署
- **輸出格式解析**: LLM 輸出格式可能不一致
  - 緩解方案: 使用結構化 prompt + JSON mode (GPT-4)
- **長歌詞處理**: 超長歌詞可能超過 context window
  - 緩解方案: 分段翻譯 + 上下文保持
- **自動特徵提取誤差**: 音節數/押韻檢測可能不準確
  - 緩解方案: 多工具交叉驗證,人工審核選項
- **評估主觀性**: 歌詞翻譯品質評估需要音樂領域知識
- **文化適應**: 不同語言的音樂表達方式差異大,直譯可能無法傳達原意

### Inference Time Optimization 策略

#### 1. **RAG (Retrieval-Augmented Generation)**
- **目標**: 從歌詞翻譯範例庫檢索相似案例,提升翻譯品質
- **實作方式**:
  ```python
  # 建立歌詞翻譯範例向量庫
  from langchain.vectorstores import FAISS
  from langchain.embeddings import OpenAIEmbeddings

  # 檢索相似翻譯範例
  similar_examples = vector_store.similarity_search(
      query=source_lyrics,
      k=3,  # 檢索 3 個最相似範例
      filter={'language_pair': f'{source_lang}-{target_lang}'}
  )

  # 將範例注入 prompt
  prompt = build_prompt_with_examples(
      source_lyrics=lyrics,
      constraints=constraints,
      retrieved_examples=similar_examples
  )
  ```
- **優勢**:
  - 動態提供高品質翻譯範例
  - 適應不同曲風和語言對
  - 無需 fine-tuning

#### 2. **ICL (In-Context Learning) Few-shot** (結合 structured output)
- **目標**: 在 prompt 中提供 2-5 個高品質翻譯範例
- **範例結構** (JSON 格式):
  ```python
  few_shot_examples = [
      {
          "source": "Let it go, let it go / Can't hold it back anymore",
          "source_lang": "English",
          "target_lang": "Chinese",
          "constraints": {
              "syllable_counts": [7, 8],
              "rhyme_scheme": "AA",
              "pause_positions": [3, 7]
          },
          "translation": {
              "translated_lines": ["放開手 放開手", "再無法 被囚禁困獸"],
              "syllable_counts": [7, 8],
              "rhyme_endings": ["手", "獸"],
              "reasoning": "保持原文簡潔有力的特點,使用疊詞'放開手'對應'let it go',押韻使用ou韻"
          }
      },
      {
          "source": "...",
          ...
      }
  ]

  # 注入 prompt
  prompt = f"""
  以下是高品質歌詞翻譯範例 (JSON 格式):

  {json.dumps(few_shot_examples, ensure_ascii=False, indent=2)}

  現在請翻譯以下歌詞,同樣以 JSON 格式輸出:

  原文: {source_lyrics}
  約束: {constraints}
  """
  ```

- **動態 ICL**: 根據輸入歌詞的特徵 (長度、曲風、押韻) 動態選擇最相關範例
- **結構化範例的好處**:
  - ✅ LLM 更容易理解輸出格式
  - ✅ 自動驗證範例的正確性
  - ✅ 易於管理和更新範例庫

#### 3. **Chain-of-Thought (CoT) Prompting** (結合 structured output)
- **目標**: 引導 LLM 逐步推理翻譯過程
- **Structured CoT Schema**:
  ```python
  class CoTTranslation(BaseModel):
      # 步驟 1: 理解
      meaning_analysis: str = Field(description="原文核心意義和情感分析")

      # 步驟 2: 約束分析
      constraint_analysis: dict = Field(
          description="音樂約束分析",
          example={
              "syllables": "需要7+8音節",
              "rhyme": "AA押韻方案",
              "pauses": "第3和第7位置需要詞邊界"
          }
      )

      # 步驟 3: 關鍵詞構思
      keyword_ideas: list[str] = Field(description="符合約束的關鍵詞候選")

      # 步驟 4: 完整譯文
      translated_lines: list[str] = Field(description="最終翻譯")

      # 步驟 5: 自我驗證
      self_verification: dict = Field(
          description="約束滿足驗證",
          example={
              "length_check": "✓ 7+8音節",
              "rhyme_check": "✓ 手/獸 AA押韻",
              "boundary_check": "✓ 詞邊界正確"
          }
      )

      # 元資料
      syllable_counts: list[int]
      rhyme_endings: list[str]

  # 使用 CoT
  translation = client.chat.completions.create(
      model="gpt-4-turbo",
      response_model=CoTTranslation,
      messages=[
          {"role": "system", "content": "請逐步推理翻譯過程"},
          {"role": "user", "content": prompt}
      ]
  )

  # 可檢視推理過程
  print(translation.meaning_analysis)
  print(translation.constraint_analysis)
  print(translation.keyword_ideas)
  print(translation.self_verification)
  ```

- **Prompt 設計**:
  ```markdown
  請按以下步驟翻譯,並以 JSON 格式輸出每個步驟的結果:

  步驟 1: 理解原文的核心意義和情感
  步驟 2: 分析音樂約束 (音節數、押韻、停頓)
  步驟 3: 構思符合約束的關鍵詞
  步驟 4: 組裝完整譯文
  步驟 5: 驗證是否滿足所有約束

  原文: {source_lyrics}
  約束: {constraints}
  ```

- **優勢**:
  - ✅ 提升複雜約束的遵守率
  - ✅ 可追蹤推理過程
  - ✅ 易於 debug 和改進
  - ✅ LLM 自我驗證提升準確度

#### 4. **Self-Consistency + Sampling**
- **目標**: 生成多個候選翻譯,選擇最佳結果
- **實作方式**:
  ```python
  # 生成 5 個候選翻譯 (使用不同 temperature)
  candidates = []
  for temp in [0.6, 0.7, 0.8, 0.9, 1.0]:
      translation = llm.generate(
          prompt=prompt,
          temperature=temp,
          n=1
      )
      candidates.append(translation)

  # 評分機制
  best_translation = rank_and_select(
      candidates=candidates,
      constraints=constraints,
      scoring_metrics=['constraint_satisfaction', 'fluency', 'semantic_similarity']
  )
  ```
- **優勢**: 降低單次生成的隨機性,提升穩定性

#### 5. **Iterative Refinement (Self-Critique)**
- **目標**: LLM 自我評估並改進翻譯
- **流程**:
  ```markdown
  # Round 1: 初次翻譯
  LLM: 生成翻譯 v1

  # Round 2: 自我評估
  Prompt: "請檢查以下翻譯是否滿足約束: [列出約束]
          翻譯: {translation_v1}
          問題: ..."

  LLM: 識別問題 (例如: 音節數不對)

  # Round 3: 改進
  Prompt: "請根據以下問題改進翻譯: {identified_issues}"

  LLM: 生成改進版 v2
  ```

#### 6. **Hybrid: RAG + CoT + Self-Consistency**
- **完整流程**:
  ```
  Input Lyrics
       ↓
  1. RAG 檢索相似範例
       ↓
  2. CoT Prompt (含範例)
       ↓
  3. 生成 N 個候選 (Self-Consistency)
       ↓
  4. 自動評分+排序
       ↓
  5. (可選) Iterative Refinement
       ↓
  Best Translation
  ```

### 推薦實作優先順序
1. **Phase 1** (最小可行方案): Zero-shot + 基礎約束 prompt
2. **Phase 2** (品質提升): 加入 ICL (手動精選 3-5 個範例)
3. **Phase 3** (自動化): 實作 RAG (動態檢索範例)
4. **Phase 4** (穩定性): 加入 Self-Consistency (多候選評分)
5. **Phase 5** (進階): CoT + Iterative Refinement

### 後續研究方向
- **多智能體協作**: 分別處理翻譯、押韻、潤飾的專門 agent
- **強化學習優化**: 基於人類反饋優化 prompt 策略
- **跨語言韻腳遷移**: 研究不同語言間押韻模式的對應關係
- **音樂領域特化**: 針對不同曲風 (流行、搖滾、民謠) 的翻譯策略
- **實時協作翻譯**: 人機協作界面,LLM 提供建議,人類微調

### 參考文獻
- **Ou et al. (2023).** "Songs Across Borders: Singable and Controllable Neural Lyric Translation." *ACL 2023*.
  - [論文連結](https://aclanthology.org/2023.acl-long.27.pdf)
  - [程式碼](https://github.com/Sonata165/ControllableLyricTranslation)
- **Low, Peter (2003).** "Singable translations of songs." *Perspectives: Studies in Translatology*, 11(2):87–103.
  - 提出 "Pentathlon Principle" 理論基礎
