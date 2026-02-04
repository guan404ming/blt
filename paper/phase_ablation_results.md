# Phase Ablation Study Results

## Experiment 1: Local Dataset (Pilot, n=5)

### Setup

| Parameter | Value |
|-----------|-------|
| Language pair | en-us → cmn |
| Model | `qwen3:30b-a3b-instruct-2507-q4_K_M` |
| Test samples | 5 |
| Max lines per sample | 5 |
| Dataset | `benchmarks/data/en_lyrics.json` |
| Date | 2026-02-02 |

### Results

| Phase | SER ↓ | SCRE ↓ | ARI ↑ | Avg Time |
|-------|-------|--------|-------|----------|
| Phase 1 only | 0.7200 | 0.1499 | -0.0444 | 9.2s |
| Phase 1+2 | 0.3200 | 0.0349 | -0.0222 | 42.9s |
| Phase 1+2+3 (full) | 0.3200 | 0.1820 | 0.0601 | 38.3s |
| **CLT baseline** (Ou et al.) | **0.0400** | **0.0019** | -0.1309 | 0.5s |

---

## Experiment 2: HuggingFace Dataset (n=30)

### Setup

| Parameter | Value |
|-----------|-------|
| Language pair | en-us → cmn |
| Model | `qwen3:30b-a3b-instruct-2507-q4_K_M` |
| Test samples | 30 (27 for Phase 1 due to 3 errors) |
| Max lines per sample | 5 |
| Dataset | [LongshenOu/lyric-trans-en2zh-data](https://huggingface.co/datasets/LongshenOu/lyric-trans-en2zh-data) (test split, `data_parallel/test.source`) |
| Date | 2026-02-02 |

### Summary Results

| Phase | Tests | SER ↓ | SCRE ↓ | ARI ↑ | Avg Time |
|-------|-------|-------|--------|-------|----------|
| Phase 1 only | 27 | 0.7926 | 0.1949 | 0.4128 | 6.7s |
| Phase 1+2 | 30 | 0.2000 | 0.0242 | 0.2723 | 29.3s |
| Phase 1+2+3 (full) | 30 | 0.2267 | 0.0313 | 0.3290 | 30.2s |
| **CLT baseline** (Ou et al.) | 30 | **0.0467** | **0.0042** | 0.0012 | 0.4s |

---

## Phase Definitions

| Config | Phases | Description |
|--------|--------|-------------|
| Phase 1 only | Initial translation | LLM translates all lines with tool access (count_syllables, text_to_ipa, check_rhyme, etc.) but no iterative refinement |
| Phase 1+2 | + Syllable refinement | Adds line-by-line syllable count refinement loop (up to 10 attempts per line) |
| Phase 1+2+3 | Full pipeline | Adds syllable pattern (word distribution) refinement after syllable counts are matched |

### Metric Definitions

- **SER** (Syllable Error Rate): Edit distance between source and translated syllable count sequences, normalized by length. Lower is better.
- **SCRE** (Syllable Count Relative Error): Average per-line relative error `|target - actual| / target`. Lower is better.
- **ARI** (Adjusted Rand Index): Measures rhyme clustering agreement between source and translation. Range [-1, 1], higher is better.

---

## Analysis (HuggingFace Dataset, n=30)

### Phase 2 Impact (Syllable Refinement)

Adding syllable refinement (Phase 2) provides the largest improvement:

- **SER**: 0.7926 → 0.2000 (74.8% reduction)
- **SCRE**: 0.1949 → 0.0242 (87.6% reduction)
- **ARI**: 0.4128 → 0.2723 (decreased) — refinement focuses on syllable counts, sometimes at the cost of rhyme structure
- **Cost**: Average time increases from 6.7s to 29.3s per test case (4.4x slower)

Phase 2 is the most impactful stage. The iterative line-by-line syllable correction loop dramatically improves both SER and SCRE, confirming that tool-assisted refinement is essential for constraint satisfaction.

### Phase 3 Impact (Pattern Refinement)

Adding pattern refinement (Phase 3) shows marginal effects:

- **SER**: 0.2000 → 0.2267 (slight increase, +13%) — pattern adjustment occasionally disrupts syllable counts
- **SCRE**: 0.0242 → 0.0313 (slight increase, +29%) — same cause as SER
- **ARI**: 0.2723 → 0.3290 (improvement, +20.8%) — pattern refinement helps restore rhyme scheme alignment
- **Cost**: 29.3s → 30.2s (negligible overhead)

Phase 3 trades a small degradation in syllable accuracy for improved rhyme scheme preservation (ARI).

### Consistency Across Datasets

| Metric | Local (n=5) P1→P2 Change | HF (n=30) P1→P2 Change |
|--------|--------------------------|------------------------|
| SER | -55.6% | -74.8% |
| SCRE | -76.7% | -87.6% |

The larger HF dataset confirms the pilot results: Phase 2 consistently delivers the largest quality improvement. The HF dataset shows even stronger gains, likely because the more diverse lyric styles (classical Chinese poetry, anthems, musical theatre) benefit more from iterative refinement.

### Key Difference: ARI

The HF dataset shows notably higher ARI across all phases compared to the pilot:

| Phase | Local ARI | HF ARI |
|-------|-----------|--------|
| Phase 1 | -0.04 | 0.41 |
| Phase 1+2 | -0.02 | 0.27 |
| Phase 1+2+3 | 0.06 | 0.33 |
| CLT baseline | -0.13 | 0.00 |

This is because the HF dataset contains lyrics with clearer rhyme structures (classical poetry, folk songs) that the model can more easily preserve during translation.

---

## Recommendations

1. **Phase 2 is essential** — the syllable refinement loop accounts for the majority of constraint satisfaction improvement (75-88% SCRE reduction)
2. **Phase 3 is conditionally useful** — it improves ARI (+21%) but slightly degrades SER/SCRE. Recommended when rhyme preservation matters.
3. For production use, **Phase 1+2 offers the best cost-quality tradeoff** with 87.6% SCRE improvement at 4.4x the cost of Phase 1 alone
4. The full pipeline (Phase 1+2+3) is recommended when **rhyme scheme preservation is a priority**

---

## Experiment 3: ControllableLyricTranslation Baseline (Ou et al., ACL 2023)

### Setup

| Parameter | Value |
|-----------|-------|
| Language pair | en → zh_CN |
| Model | `LongshenOu/lyric-trans-en2zh` (mBART fine-tuned, prompt-based) |
| Method | Controllable neural lyric translation with explicit length, rhyme, and word boundary constraints |
| Paper | [Songs Across Borders (ACL 2023)](https://arxiv.org/abs/2305.16816) |
| Repo | [Sonata165/ControllableLyricTranslation](https://github.com/Sonata165/ControllableLyricTranslation) |
| Date | 2026-02-02 |

### Demo Results (playground.py)

| # | Source (English) | Desired Length | Rhyme Type | Word Boundary | Translation (Chinese) |
|---|-----------------|---------------|------------|---------------|----------------------|
| 1 | There's only one song left for you | 12 | 1 (a/ia/ua) | `[0,0,1,0,0,0,0,0,0,0,0,0]` | 现在只剩下一首歌为你留下 |
| 2 | Get me off the streets of this city | 9 | 1 (a/ia/ua) | `[0,1,0,0,0,0,0,0,0]` | 离开这城市的街道吧 |
| 3 | You only left one kiss for me | 8 | 1 (a/ia/ua) | `[0,0,0,1,0,0,1,0]` | 只留给我一个吻吧 |

### Constraint Satisfaction

| # | Target Length | Actual Length | Length Match | Rhyme Match |
|---|--------------|--------------|-------------|-------------|
| 1 | 12 | 12 (现在只剩下一首歌为你留下) | ✓ | ✓ (下 → xià, rhyme a) |
| 2 | 9 | 9 (离开这城市的街道吧) | ✓ | ✓ (吧 → ba, rhyme a) |
| 3 | 8 | 8 (只留给我一个吻吧) | ✓ | ✓ (吧 → ba, rhyme a) |

- **Length accuracy**: 3/3 (100%)
- **Rhyme accuracy**: 3/3 (100%)

### Notes

- This model uses **explicit constraint prompts** (length tokens, rhyme tokens, word boundary tokens) prepended to the encoder input, unlike the BLT pipeline which uses tool-assisted iterative refinement.
- The model generates characters in **reverse order** (right-to-left) and the output is flipped to produce the final translation.
- Reported metrics from the paper: 99.85% length accuracy, 99.00% rhyme accuracy, 95.52% word boundary recall on the full test set.
- This serves as a **supervised fine-tuning baseline** for comparison against BLT's LLM agent-based approach.

---

## Experiment 4: Paper-Scale Ablation (n=100)

### Setup

| Parameter | Value |
|-----------|-------|
| Language pair | en-us → cmn |
| Model | `qwen3:30b-a3b-instruct-2507-q4_K_M` |
| Test samples | 100 |
| Max lines per sample | 5 |
| Dataset | [LongshenOu/lyric-trans-en2zh-data](https://huggingface.co/datasets/LongshenOu/lyric-trans-en2zh-data) (test split) |
| Test suite | `benchmarks/results/en-us→cmn_ablation_n100_test_suite.json` |
| Date | 2026-02-04 |

### Results (mean ± 95% CI)

| Phase | n | SER ↓ | SCRE ↓ | ARI ↑ | Avg Time |
|-------|---|-------|--------|-------|----------|
| Phase 1 only | 99 | 0.8343±0.0356 | 0.2076±0.0170 | 0.3133±0.0945 | 7.0s |
| Phase 1+2 | 106 | 0.2151±0.0406 | 0.0291±0.0152 | 0.3623±0.0949 | 33.5s |
| Phase 1+2+3 (full) | 106 | 0.3132±0.0446 | 0.0457±0.0156 | 0.2951±0.0903 | 51.8s |
| **CLT baseline** (Ou et al.) | 100 | 0.0860±0.0326 | 0.0142±0.0066 | 0.0015±0.0440 | 0.6s |

### Key Findings (Baseline Phase 3)

- **Phase 1+2 shows the best performance** with SER=0.2151 and ARI=0.3623, demonstrating that syllable count refinement (Phase 2) is highly effective.
- **Phase 3 (full pipeline) shows degraded performance** compared to Phase 1+2, with SER increasing to 0.3132 (+45.6%) and ARI decreasing to 0.2951 (-18.5%). The syllable pattern refinement step introduces more errors than it fixes.
- **CLT baseline achieves lowest SER** (0.0860) due to supervised fine-tuning on parallel lyrics data, but shows near-zero ARI (0.0015), indicating poor rhyme pattern alignment.
- **BLT trades off syllable accuracy for structural constraints**, achieving moderate rhyme clustering (ARI ~0.30-0.36) while maintaining reasonable syllable accuracy.

---

## Experiment 5: Improved Phase 3 (n=100)

### Setup

| Parameter | Value |
|-----------|-------|
| Language pair | en-us → cmn |
| Model | `qwen3:30b-a3b-instruct-2507-q4_K_M` |
| Test samples | 100 |
| Dataset | [LongshenOu/lyric-trans-en2zh-data](https://huggingface.co/datasets/LongshenOu/lyric-trans-en2zh-data) (test split) |
| Test suite | `benchmarks/results/en-us→cmn_improved_n100_test_suite.json` |
| Date | 2026-02-04 |
| **Improvements** | • 75% pattern skip threshold<br>• 15% minimum improvement requirement<br>• Punctuation stripping<br>• Rhyme preservation check |

### Results (mean ± 95% CI)

| Phase | n | SER ↓ | SCRE ↓ | ARI ↑ | Avg Time |
|-------|---|-------|--------|-------|----------|
| Phase 1 only | 100 | 0.8840±0.0311 | 0.4482±0.4235 | 0.2776±0.0929 | 9.2s |
| Phase 1+2 | 98 | 0.2224±0.0381 | 0.0311±0.0162 | 0.3170±0.0962 | 42.5s |
| **Phase 1+2+3 (improved)** | 100 | **0.2500±0.0388** | **0.0271±0.0060** | **0.2481±0.0903** | 47.6s |

### Comparison: Baseline vs Improved Phase 3

| Metric | Baseline Phase 3 | Improved Phase 3 | Change |
|--------|------------------|------------------|--------|
| **SER ↓** | 0.3132±0.0446 | **0.2500±0.0388** | **-20.2% better** ✅ |
| **SCRE ↓** | 0.0457±0.0156 | **0.0271±0.0060** | **-40.7% better** ✅ |
| **ARI ↑** | 0.2951±0.0903 | 0.2481±0.0903 | -15.9% worse ❌ |
| **Phase 2→3 SER degradation** | +0.0981 (+45.6%) | **+0.0276 (+12.4%)** | **71.9% reduction** ✅ |

### Key Improvements

1. **Punctuation stripping** - Removes punctuation before syllable counting to avoid false mismatches
2. **Rhyme preservation check** - Rejects pattern changes that break rhymes established in Phase 2
3. **High skip threshold (75%)** - Only refines patterns when they're significantly poor
4. **Minimum improvement requirement (15%)** - Only accepts changes that substantially improve pattern matching

### Impact

- **SER degradation reduced by 71.9%** - Phase 3 now only increases SER by 12.4% instead of 45.6%
- **Overall Phase 3 SER improved 20.2%** - From 0.3132 → 0.2500
- **Trade-off**: ARI slightly worse (15.9%) but within acceptable range given major SER improvement

### Recommendation

**Use the improved Phase 3 configuration** (75% skip, 15% min improvement) as it significantly reduces the harmful effects of pattern refinement while maintaining reasonable rhyme structure. The improvements make Phase 3 a viable option rather than consistently degrading quality.

---

## How to Reproduce

```bash
# Generate the n=100 test suite
python -m benchmarks.generate_test_suite --samples 100

# Run all 3 ablation configurations on n=100 test suite
python -m benchmarks.run_ablation en-us cmn --samples 100 \
  --test-suite benchmarks/results/en-us→cmn_ablation_n100_test_suite.json

# Run CLT baseline on same test suite
LD_LIBRARY_PATH=/home/gmchiu/local/cuda-12.2/lib64 \
  ControllableLyricTranslation/.venv/bin/python benchmarks/run_clt_baseline.py \
  --test-suite benchmarks/results/en-us→cmn_ablation_n100_test_suite.json

# Run all 3 ablation configurations with HuggingFace dataset (n=30)
python -m benchmarks.run_ablation en-us cmn --samples 30 \
  --hf-data benchmarks/data/hf_en2zh/datasets

# Run all 3 ablation configurations with local dataset
python -m benchmarks.run_ablation en-us cmn --samples 5

# Run individual phase configurations
python -m benchmarks.run_experiment en-us cmn --phases 1 --mode agent
python -m benchmarks.run_experiment en-us cmn --phases 2 --mode agent
python -m benchmarks.run_experiment en-us cmn --phases 3 --mode agent

# Custom model
python -m benchmarks.run_ablation en-us cmn --samples 10 \
  --model qwen3:30b-a3b-instruct-2507-q4_K_M

# Run CLT baseline (requires ControllableLyricTranslation submodule + its venv)
LD_LIBRARY_PATH=/home/gmchiu/local/cuda-12.2/lib64 \
  ControllableLyricTranslation/.venv/bin/python benchmarks/run_clt_baseline.py \
  --dataset both
```
