#!/usr/bin/env python3
"""
Run ControllableLyricTranslation (Ou et al., ACL 2023) baseline evaluation.

Evaluates the mBART-based lyric translation model on the same test data
used in BLT's phase ablation experiments, using BLT's metrics (SER, SCRE, ARI).

Usage:
    # Activate the CLT venv first, then run from the blt project root:
    LD_LIBRARY_PATH=/home/gmchiu/local/cuda-12.2/lib64 \
    python benchmarks/run_clt_baseline.py --dataset local --samples 5

    LD_LIBRARY_PATH=/home/gmchiu/local/cuda-12.2/lib64 \
    python benchmarks/run_clt_baseline.py --dataset hf --samples 30
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

# Add CLT BartFinetune to path for model imports
CLT_ROOT = Path(__file__).resolve().parent.parent / "ControllableLyricTranslation"
sys.path.insert(0, str(CLT_ROOT / "BartFinetune"))

from models.MBarts import MBartForConditionalGenerationCharLevel, MBart50TokenizerFast


# ---------------------------------------------------------------------------
# BLT-compatible metric calculation (self-contained, no BLT imports needed)
# ---------------------------------------------------------------------------

def _levenshtein(seq1: list, seq2: list) -> int:
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def calc_ser(source_syllables: list[int], target_syllables: list[int]) -> float:
    if not source_syllables:
        return 0.0
    ed = _levenshtein(source_syllables, target_syllables)
    n = max(len(source_syllables), len(target_syllables))
    return ed / n if n > 0 else 0.0


def calc_scre(source_syllables: list[int], target_syllables: list[int]) -> float:
    if not source_syllables:
        return 0.0
    errs = []
    for src, tgt in zip(source_syllables, target_syllables):
        if src == 0:
            errs.append(0.0 if tgt == 0 else 1.0)
        else:
            errs.append(abs(src - tgt) / src)
    return sum(errs) / len(errs) if errs else 0.0


def _scheme_to_labels(scheme: str) -> list[int]:
    m, labels, nxt = {}, [], 0
    for c in scheme:
        if c not in m:
            m[c] = nxt
            nxt += 1
        labels.append(m[c])
    return labels


def calc_ari(source_scheme: str, target_scheme: str) -> float:
    if not source_scheme or not target_scheme:
        return 0.0
    from sklearn.metrics import adjusted_rand_score
    sl = _scheme_to_labels(source_scheme)
    tl = _scheme_to_labels(target_scheme)
    if len(sl) != len(tl):
        ml = max(len(sl), len(tl))
        nxt = max(max(sl), max(tl)) + 1
        if len(sl) < ml:
            sl += list(range(nxt, nxt + ml - len(sl)))
        if len(tl) < ml:
            tl += list(range(nxt, nxt + ml - len(tl)))
    return adjusted_rand_score(sl, tl)


# ---------------------------------------------------------------------------
# Syllable counting (BLT-compatible)
# ---------------------------------------------------------------------------

def count_syllables_en(text: str) -> int:
    """Count English syllables via espeak IPA (same as BLT analyzer)."""
    import subprocess
    cleaned = re.sub(r"[,;.!?，。；！？、\s]+", "", text)
    if not cleaned:
        return 0
    try:
        result = subprocess.run(
            ["espeak-ng", "--ipa=3", "-q", "--stdin", "-v", "en-gb"],
            input=cleaned, capture_output=True, text=True, timeout=5,
        )
        ipa = result.stdout.strip()
    except Exception:
        # Fallback: rough vowel-group count
        return len(re.findall(r"[aeiouy]+", cleaned.lower()))

    diphthong_pat = r"(?:aɪ|eɪ|ɔɪ|aʊ|oʊ|ɪə|eə|ʊə|aɪə|aʊə|[iɪeɛæaäɑɒɔoʊuʉɨəɜɞʌyøœɶɐɚɝɯ][\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F]*ː?)"
    return len(re.findall(diphthong_pat, ipa))


def count_syllables_zh(text: str) -> int:
    """Count Chinese syllables (= number of characters after removing punctuation)."""
    cleaned = re.sub(r"[,;.!?，。；！？、\s]+", "", text)
    return len(cleaned)


# ---------------------------------------------------------------------------
# Rhyme scheme detection (BLT-compatible)
# ---------------------------------------------------------------------------

def detect_rhyme_scheme_zh(lines: list[str]) -> str:
    """Detect rhyme scheme for Chinese lines using pypinyin finals."""
    from pypinyin import pinyin, Style

    endings = []
    for line in lines:
        line = line.strip()
        if not line:
            endings.append("")
            continue
        finals = pinyin(line, style=Style.FINALS, strict=False)
        if finals and finals[-1]:
            endings.append(finals[-1][0])
        else:
            endings.append("")
    return _endings_to_scheme(endings)


def detect_rhyme_scheme_en(lines: list[str]) -> str:
    """Detect rhyme scheme for English lines using espeak IPA."""
    import subprocess

    ipa_pat = r"[iɪeɛæaäɑɒɔoʊuʉɨəɜɞʌyøœɶɐɚɝɯ][\u0300-\u036F\u1AB0-\u1AFF\u1DC0-\u1DFF\u20D0-\u20FF\uFE20-\uFE2F]*"
    endings = []
    for line in lines:
        line = line.strip()
        if not line:
            endings.append("")
            continue
        try:
            result = subprocess.run(
                ["espeak-ng", "--ipa=3", "-q", "--stdin", "-v", "en-gb"],
                input=line, capture_output=True, text=True, timeout=5,
            )
            ipa = result.stdout.strip()
            matches = list(re.finditer(ipa_pat, ipa))
            if matches:
                endings.append(ipa[matches[-1].start():])
            else:
                endings.append("")
        except Exception:
            endings.append("")
    return _endings_to_scheme(endings)


def _endings_to_scheme(endings: list[str]) -> str:
    scheme = []
    label_map = {}
    next_label = 0
    for e in endings:
        if not e:
            scheme.append(chr(ord("A") + next_label))
            next_label += 1
            continue
        matched = False
        for prev_e, lbl in label_map.items():
            if e == prev_e or e in prev_e or prev_e in e:
                scheme.append(lbl)
                matched = True
                break
        if not matched:
            lbl = chr(ord("A") + next_label)
            next_label += 1
            label_map[e] = lbl
            scheme.append(lbl)
    return "".join(scheme)


# ---------------------------------------------------------------------------
# Dataset loading (matches BLT experiment data)
# ---------------------------------------------------------------------------

def load_local_dataset(data_dir: str, num_samples: int = 5, max_lines: int = 5) -> list[dict]:
    """Load test cases from local en_lyrics.json (same as BLT Experiment 1)."""
    filepath = Path(data_dir) / "en_lyrics.json"
    with open(filepath, "r", encoding="utf-8") as f:
        songs = json.load(f)

    test_cases = []
    for i, song in enumerate(songs):
        if song.get("line_count", 0) < 4:
            continue
        raw = [l.strip() for l in song["lyrics"].split("\n") if l.strip()]
        # Filter non-English lines
        lines = []
        for l in raw:
            en_chars = len(re.findall(r"[a-zA-Z]", l))
            total = en_chars + len(re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", l))
            if total == 0 or en_chars / max(total, 1) > 0.5 or total == 0:
                lines.append(l)
        if len(lines) < 4:
            continue
        lines = lines[:max_lines]
        test_cases.append({
            "id": f"en-us→cmn_{i+1:03d}",
            "source_lines": lines,
            "source_lang": "en-us",
            "target_lang": "cmn",
            "metadata": {
                "song_title": song.get("song_title", "Unknown"),
                "artist_name": song.get("artist_name", "Unknown"),
                "line_count": len(lines),
            },
        })
        if len(test_cases) >= num_samples:
            break
    return test_cases


def load_hf_dataset(data_dir: str, num_samples: int = 30, max_lines: int = 5) -> list[dict]:
    """Load test cases from HF parallel data (same as BLT Experiment 2)."""
    source_file = Path(data_dir) / "data_parallel" / "test.source"
    with open(source_file, "r", encoding="utf-8") as f:
        all_lines = [l.strip() for l in f if l.strip()]

    test_cases = []
    for i in range(0, len(all_lines), max_lines):
        chunk = all_lines[i:i + max_lines]
        if len(chunk) < 2:
            continue
        test_cases.append({
            "id": f"en-us→cmn_hf_{i // max_lines + 1:03d}",
            "source_lines": chunk,
            "source_lang": "en-us",
            "target_lang": "cmn",
            "metadata": {
                "song_title": f"HF chunk {i // max_lines + 1}",
                "artist_name": "lyric-trans-en2zh-data",
                "line_count": len(chunk),
            },
        })
    return test_cases[:num_samples]


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def translate_lines(
    model, tokenizer, device: str,
    lines: list[str], target_lengths: list[int],
) -> list[str]:
    """
    Translate English lines to Chinese using ControllableLyricTranslation model.

    Sets length constraint to match source syllable count.
    Uses rhyme type 0 and zero boundary (no specific rhyme/boundary constraints).
    """
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "zh_CN"
    batch_size = len(lines)

    encoded = tokenizer(lines, return_tensors="pt", padding=True).to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Length constraint tokens
    tgt_lens = [f"len_{l}" for l in target_lengths]
    t1 = tokenizer(tgt_lens, add_special_tokens=False, return_tensors="pt",
                    max_length=1, padding=False, truncation=True)
    tgt_lens_ids = t1["input_ids"].to(device)
    attn_len = t1["attention_mask"].to(device)

    # Rhyme constraint (type 0 = neutral)
    tgt_rhymes = [f"rhy_0" for _ in range(batch_size)]
    t2 = tokenizer(tgt_rhymes, add_special_tokens=False, return_tensors="pt",
                    max_length=1, padding=False, truncation=True)
    tgt_rhymes_ids = t2["input_ids"].to(device)

    # Boundary constraint (all zeros, length matching target)
    boundaries = [[0] * l for l in target_lengths]
    tgt_stress = ["".join([f"str_{i}" for i in b[::-1]]) for b in boundaries]
    t3 = tokenizer(tgt_stress, return_tensors="pt", add_special_tokens=False, padding=True)
    tgt_stress_ids = t3["input_ids"].to(device)
    attn_str = t3["attention_mask"].to(device)
    pad_bit = 20 - tgt_stress_ids.shape[1]
    if pad_bit > 0:
        tgt_stress_ids = F.pad(tgt_stress_ids, (0, pad_bit), value=1).to(device)
        attn_str = F.pad(attn_str, (0, pad_bit), value=1).to(device)
    elif pad_bit < 0:
        tgt_stress_ids = tgt_stress_ids[:, :20]
        attn_str = attn_str[:, :20]

    # Concat constraints with encoder input
    input_ids = torch.cat((tgt_lens_ids, tgt_stress_ids, input_ids), dim=1).to(device)
    attention_mask = torch.cat((attn_len, attn_str, attention_mask), dim=1).to(device)

    # Decoder input: rhyme + start token
    decoder_input_ids = torch.zeros(size=(batch_size, 2), dtype=torch.long).to(device)
    decoder_input_ids[:, 0] = tgt_rhymes_ids.squeeze()
    decoder_input_ids[:, 1] = 2  # decoder_start_token_id

    # Generate
    generated = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        num_beams=5,
        max_length=36,
        forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"],
    )

    # Decode (model generates reversed, so flip)
    translations = []
    for tokens in tokenizer.batch_decode(generated, skip_special_tokens=True):
        translations.append(tokens[::-1])
    return translations


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(test_cases: list[dict], model, tokenizer, device: str) -> dict:
    """Run model on all test cases and compute BLT metrics."""
    all_results = []
    total_ser, total_scre, total_ari = 0.0, 0.0, 0.0
    total_time = 0.0

    for tc in test_cases:
        source_lines = tc["source_lines"]

        # Count source syllables (same as BLT)
        source_syllables = [count_syllables_en(l) for l in source_lines]
        # Clamp to [1, 35] for model compatibility (max_length=36)
        target_lengths = [max(1, min(s, 35)) for s in source_syllables]

        # Translate
        t0 = time.time()
        try:
            translated = translate_lines(model, tokenizer, device, source_lines, target_lengths)
        except Exception as e:
            print(f"  Error on {tc['id']}: {e}")
            translated = ["" for _ in source_lines]
        elapsed = time.time() - t0

        # Count target syllables
        target_syllables = [count_syllables_zh(t) for t in translated]

        # Rhyme schemes
        source_scheme = detect_rhyme_scheme_en(source_lines)
        target_scheme = detect_rhyme_scheme_zh(translated)

        # Metrics
        ser = calc_ser(source_syllables, target_syllables)
        scre = calc_scre(source_syllables, target_syllables)
        ari = calc_ari(source_scheme, target_scheme)

        total_ser += ser
        total_scre += scre
        total_ari += ari
        total_time += elapsed

        result = {
            "test_id": tc["id"],
            "source_lines": source_lines,
            "translated_lines": translated,
            "source_syllables": source_syllables,
            "target_syllables": target_syllables,
            "source_rhyme_scheme": source_scheme,
            "target_rhyme_scheme": target_scheme,
            "metrics": {"ser": ser, "scre": scre, "ari": ari},
            "time_seconds": elapsed,
            "metadata": tc.get("metadata", {}),
        }
        all_results.append(result)
        print(f"  {tc['id']}: SER={ser:.4f}  SCRE={scre:.4f}  ARI={ari:.4f}  time={elapsed:.1f}s")

    n = len(all_results)
    avg_metrics = {
        "ser": total_ser / n if n else 0,
        "scre": total_scre / n if n else 0,
        "ari": total_ari / n if n else 0,
        "avg_time_seconds": total_time / n if n else 0,
    }
    return {"results": all_results, "avg_metrics": avg_metrics, "total_tests": n}


def main():
    parser = argparse.ArgumentParser(description="Run CLT baseline with BLT metrics")
    parser.add_argument("--dataset", choices=["local", "hf", "both"], default="both",
                        help="Dataset to evaluate on")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples (default: 5 for local, 30 for hf)")
    parser.add_argument("--max-lines", type=int, default=5,
                        help="Max lines per test case")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--model-path", default=None,
                        help="Path to model (default: auto-detect from model_cache)")
    args = parser.parse_args()

    # Find model
    if args.model_path is None:
        cache_dir = CLT_ROOT / "model_cache"
        snapshots = list(cache_dir.glob("models--*/snapshots/*"))
        if snapshots:
            args.model_path = str(snapshots[0])
        else:
            print("Error: No model found. Download first.")
            sys.exit(1)

    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")

    # Load model
    print("Loading model...")
    model = MBartForConditionalGenerationCharLevel.from_pretrained(args.model_path)
    tokenizer = MBart50TokenizerFast.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()
    print("Model loaded.")

    project_root = Path(__file__).resolve().parent.parent

    datasets_to_run = []
    if args.dataset in ("local", "both"):
        n = args.samples if args.samples is not None else 5
        datasets_to_run.append(("local", n))
    if args.dataset in ("hf", "both"):
        n = args.samples if args.samples is not None else 30
        datasets_to_run.append(("hf", n))

    all_outputs = {}
    for ds_name, n_samples in datasets_to_run:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name} (n={n_samples})")
        print(f"{'=' * 60}")

        if ds_name == "local":
            test_cases = load_local_dataset(
                str(project_root / "benchmarks" / "data"),
                num_samples=n_samples, max_lines=args.max_lines,
            )
        else:
            test_cases = load_hf_dataset(
                str(project_root / "benchmarks" / "data" / "hf_en2zh" / "datasets"),
                num_samples=n_samples, max_lines=args.max_lines,
            )

        print(f"Test cases: {len(test_cases)}")
        output = run_evaluation(test_cases, model, tokenizer, args.device)
        all_outputs[ds_name] = output

        m = output["avg_metrics"]
        print(f"\n--- {ds_name} Average (n={output['total_tests']}) ---")
        print(f"  SER:  {m['ser']:.4f}")
        print(f"  SCRE: {m['scre']:.4f}")
        print(f"  ARI:  {m['ari']:.4f}")
        print(f"  Time: {m['avg_time_seconds']:.1f}s")

    # Save JSON results
    results_dir = project_root / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for ds_name, output in all_outputs.items():
        outfile = results_dir / f"clt_baseline_{ds_name}.json"
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved: {outfile}")

    # Print summary table
    print(f"\n{'=' * 75}")
    print("CLT BASELINE SUMMARY")
    print(f"{'=' * 75}")
    print(f"{'Dataset':<20} {'Tests':>6} {'SER ↓':>10} {'SCRE ↓':>10} {'ARI ↑':>10} {'Avg Time':>10}")
    print(f"{'─' * 75}")
    for ds_name, output in all_outputs.items():
        m = output["avg_metrics"]
        print(f"{ds_name:<20} {output['total_tests']:>6} {m['ser']:>10.4f} {m['scre']:>10.4f} {m['ari']:>10.4f} {m['avg_time_seconds']:>9.1f}s")


if __name__ == "__main__":
    main()
