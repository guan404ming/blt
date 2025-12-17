# Evaluation Metrics

We adopt three core evaluation metrics to evaluate lyrics translation:

## 1. Syllable Error Rate (SER) ↓ (lower is better)

**Definition:**
```
SER(A, T) = EditDistance(A, T) / n
```

**Range:** [0, ∞)

**Description:**
Measures syllable-count sequence error using Levenshtein distance. Each operation (insert, delete, substitute) has a cost of 1.

**Use case:** Ensures the translated lyrics maintain consistent syllable counts with the target.

---

## 2. Syllable Count Relative Error (SCRE) ↓ (lower is better)

**Definition:**
```
SCRE = (1/n) × Σ |syllables_target(i) - syllables_actual(i)| / syllables_target(i)
```

Where:
- `n` = number of lines
- `syllables_target(i)` = syllable count of target line i
- `syllables_actual(i)` = syllable count of actual/predicted line i

**Range:** [0, ∞)

**Description:**
Measures the average relative error in syllable counts per line. Each line's error is normalized by its target syllable count, providing a percentage deviation metric. Handles zero syllable counts gracefully.

**Use case:** Complements SER by measuring percentage-based accuracy, useful for comparing translations of varying lengths.

---

## 3. Rhyme Preservation Rate (RPR) ↑ (higher is better)

**Definition:**
```
RPR = (# matched end-of-line rhymes) / (total number of target lines)
```

**Description:**
Measures the fraction of target end-of-line rhymes that are preserved in the predicted sequence, allowing matches at different positions.

**Range:** [0, 1]

**Example:**

Target rhyme scheme: ABACCA
Actual rhyme scheme: AAABBC

Matched indices: 1, 3, 4, 5
RPR = 4 / 6 ≈ 0.667

**Use case:** Ensures the translated lyrics maintain the rhyme scheme of the original.
