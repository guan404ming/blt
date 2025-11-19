# 研究規劃：基於結構化 In-Context Learning 的 Text-to-Music 生成

## 一、研究背景與動機

### 1.1 問題陳述
現有 Text-to-Music (T2M) 模型在以下方面存在不足：
- 長篇音樂結構的一致性
- 高階音樂性推理（如調式選擇、和弦進行邏輯）
- 對複雜文本提示的細緻映射

### 1.2 研究假設
透過結構化的 Few-Shot In-Context Learning，模型能從多個 `(text, encoded_audio)` 範例中學習：
- 文本描述與音樂特徵間的複雜映射規則
- 風格一致性的保持
- 音樂理論層面的推理模式

---

## 二、研究目標

### 2.1 主要目標
開發一套基於結構化 ICL 的 T2M 框架，使模型能透過少量範例學習文本到音樂的映射模式。

### 2.2 具體目標
1. 設計有效的序列化輸入格式
2. 探索最佳的範例數量與選擇策略
3. 評估 ICL 對音樂生成品質的提升效果
4. 比較 Zero-shot vs Few-shot 性能差異

---

## 三、技術方法

### 3.1 輸入序列設計
```
[Text₁ Tokens] [Encoded_Audio₁ Tokens] <SEP>
[Text₂ Tokens] [Encoded_Audio₂ Tokens] <SEP>
...
[Textₙ Tokens] [Encoded_Audioₙ Tokens] <SEP>
[Target_Prompt Tokens]
```

### 3.2 實作路徑

#### 路徑 A：零樣本推理（適用於大型模型）
- 直接測試現有大型音樂生成模型的 ICL 能力
- 無需額外訓練

#### 路徑 B：輕量級微調
- 基於 MusicGen / AudioLDM 等現有模型
- 微調目標：學習辨識結構化輸入格式
- 訓練數據：構建 `(examples, target)` 配對數據集

### 3.3 音訊編碼方案
- **選項 1**: EnCodec tokens (離散)
- **選項 2**: CLAP embeddings (連續)
- **選項 3**: 混合表示

---

## 四、實驗設計

### 4.1 數據集
| 數據集 | 用途 | 規模 |
|--------|------|------|
| MusicCaps | 訓練/評估 | ~5.5K |
| Song Describer | 補充訓練 | ~1.1K |
| 自建風格數據集 | 特定風格測試 | 待定 |

### 4.2 實驗組別
1. **Baseline**: 標準 T2M (Zero-shot)
2. **Single-ICL**: 1 個參考範例
3. **Multi-ICL**: 2-5 個參考範例
4. **Structured-ICL**: 帶有明確規則的範例組

### 4.3 評估指標
- **客觀指標**:
  - FAD (Fréchet Audio Distance)
  - KL Divergence
  - CLAP Score (文本-音訊對齊)

- **主觀指標**:
  - 音樂品質 MOS
  - 風格一致性評分
  - 提示遵循度

### 4.4 消融實驗
- 範例數量的影響 (1, 2, 3, 5)
- 範例選擇策略（隨機 vs 語義相似 vs 風格多樣）
- 分隔符設計的影響

---

## 五、預期研究貢獻

1. **方法論貢獻**: 首個將結構化 ICL 應用於 T2M 的系統性研究
2. **技術貢獻**: 開源的 ICL-T2M 框架與訓練代碼
3. **實證貢獻**: 全面的消融實驗與基準測試結果
4. **理論貢獻**: 理解 T2M 模型如何從範例中學習音樂規則

---

## 六、研究階段

### 階段一：基礎建設
- 文獻回顧與相關工作整理
- 數據集準備與預處理
- 基礎模型選擇與環境搭建

### 階段二：核心開發
- 設計序列化輸入格式
- 實作 ICL 推理 pipeline
- 初步實驗驗證可行性

### 階段三：微調與優化
- 設計微調策略
- 訓練模型辨識 ICL 格式
- 超參數調整

### 階段四：全面評估
- 完整實驗執行
- 消融研究
- 用戶研究

### 階段五：論文撰寫
- 結果分析與整理
- 論文撰寫與投稿

---

## 七、潛在挑戰與解決方案

| 挑戰 | 解決方案 |
|------|----------|
| 序列長度過長 | 使用高效壓縮編碼；限制範例數量 |
| 訓練數據不足 | 數據增強；遷移學習 |
| 評估困難 | 結合客觀指標與人工評估 |
| 計算資源限制 | 使用 LoRA 等參數高效微調方法 |

---

## 八、消費級 GPU 優化設計

### 8.1 目標硬體規格
- **開發環境**: RTX 5070 Ti (16GB VRAM)
- **主要目標**: RTX 3090 / 4090 (24GB VRAM)
- **最低要求**: RTX 3080 / 4080 (10-16GB VRAM)
- 存儲: ~200GB（數據集與模型）

### 8.2 記憶體優化策略

#### 8.2.1 模型選擇
| 模型 | 參數量 | VRAM 需求 | 推薦度 |
|------|--------|-----------|--------|
| MusicGen-small | 300M | ~4GB | 開發測試 |
| MusicGen-medium | 1.5B | ~8GB | 主要實驗 |
| MusicGen-large | 3.3B | ~16GB | 最終評估 |

#### 8.2.2 參數高效微調 (PEFT)
- **LoRA** (Low-Rank Adaptation)
  - Rank: 8-32
  - Alpha: 16-64
  - 目標層: attention layers
  - VRAM 節省: ~70%

- **QLoRA** (Quantized LoRA)
  - 4-bit 量化基礎模型
  - LoRA 適配器保持 FP16
  - VRAM 需求: ~6GB (medium model)

#### 8.2.3 序列長度管理
```python
# ICL 範例壓縮策略
max_audio_tokens = 750      # 原本 1500，壓縮 50%
max_text_tokens = 128       # 精簡文本描述
max_icl_examples = 2-3      # 限制範例數量
total_sequence = ~3000      # 可在 24GB 內運行
```

#### 8.2.4 梯度檢查點 (Gradient Checkpointing)
- 以計算換記憶體
- VRAM 節省: ~40%
- 訓練時間增加: ~20%

### 8.3 訓練優化

#### 8.3.1 混合精度訓練
```python
# 使用 bfloat16/float16
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast(dtype=torch.bfloat16):
    loss = model(inputs)
```

#### 8.3.2 批次大小策略
| GPU | Batch Size | Gradient Accumulation | Effective Batch |
|-----|------------|----------------------|-----------------|
| RTX 4090 (24GB) | 2 | 8 | 16 |
| RTX 3080 (10GB) | 1 | 16 | 16 |

#### 8.3.3 優化器選擇
- **8-bit Adam** (bitsandbytes)
  - 記憶體節省: ~50% optimizer states
- **Adafactor**
  - 無需儲存 momentum
  - 適合極低記憶體環境

### 8.4 推理優化

#### 8.4.1 KV Cache 優化
- 使用 Flash Attention 2
- Sliding window attention (如適用)

#### 8.4.2 模型量化
```python
# 推理時 4-bit 量化
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
```

#### 8.4.3 音訊編碼壓縮
- EnCodec: 使用較高壓縮率 (1.5 kbps vs 6 kbps)
- Semantic tokens only (捨棄 acoustic tokens 在 ICL 階段)

### 8.5 分散式訓練 (多卡消費級)

#### 8.5.1 DeepSpeed ZeRO Stage 2
```python
# 適合 2-4 張 RTX 3090/4090
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    }
}
```

#### 8.5.2 FSDP (Fully Sharded Data Parallel)
- PyTorch 原生支援
- 適合同質 GPU 環境

### 8.6 實際 VRAM 估算

| 配置 | 模型 | 方法 | 預估 VRAM |
|------|------|------|-----------|
| 最低 | MusicGen-small | QLoRA + GC | ~6GB |
| 推薦 | MusicGen-medium | LoRA + GC + AMP | ~12GB |
| 完整 | MusicGen-large | LoRA + GC + AMP | ~20GB |

*GC = Gradient Checkpointing, AMP = Automatic Mixed Precision*

### 8.7 軟體/框架
- PyTorch 2.0+ (compile 優化)
- Transformers + PEFT
- bitsandbytes (量化)
- Flash Attention 2
- AudioCraft (MusicGen)
- Weights & Biases (實驗追蹤)

### 8.8 推薦開發流程
1. **原型開發**: MusicGen-small + QLoRA on RTX 3080
2. **主要實驗**: MusicGen-medium + LoRA on RTX 4090
3. **最終驗證**: MusicGen-large + LoRA on 2x RTX 4090

---

## 九、預期產出

1. 頂級會議/期刊論文（目標：ICML / NeurIPS / ISMIR）
2. 開源代碼與預訓練模型
3. 公開評測基準
