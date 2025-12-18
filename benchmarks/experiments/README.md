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

## 3. Adjusted Rand Index (ARI) ↑ (higher is better)

**Definition:**

```
ARI = (RI - E[RI]) / (max(RI) - E[RI])
```

Where RI is the Rand Index comparing rhyme clustering.

**Description:**
Measures how well the rhyme **clustering** (which lines rhyme together) is preserved, independent of label names. Unlike position-by-position comparison, ARI compares the structure of rhyming relationships.

**Range:** [-1, 1]

- `+1.0` = Perfect rhyme clustering agreement
- `0.0` = Random/chance clustering
- `-1.0` = Complete disagreement

**Use case:** Ensures the translated lyrics maintain the rhyme structure of the original.
