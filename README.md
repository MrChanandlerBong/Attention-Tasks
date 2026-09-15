# Attention Internals — Postman AI/ML Recruitment (25 Batch)

Two deep dives into how attention actually works under the hood, and what breaks when you mess with it.

---

## Task 1 — Sparse Attention from Scratch

Built three attention mechanisms by hand (no `F.scaled_dot_product_attention`, no FlashAttention, just matmuls and masks) and compared them on everything that matters: correctness, edge cases, speed, and whether they actually hurt a language model.

### Files

| File | What it does |
|------|-------------|
| `dense_attn.py` | Vanilla scaled dot-product attention. The reference. |
| `sliding_window_attn.py` | Each token only sees the last `window_size` tokens. |
| `global_window_attn.py` | Sliding window + designated global tokens that act as info hubs — they see everything, everything sees them. |
| `sanity.py` | Correctness harness — checks sparse output against dense-with-the-same-mask (not raw dense, because softmax normalization differs). |
| `boom.py` | Demonstrates the NaN failure mode when a query has zero valid keys, and shows the fix. |
| `speedrun.py` | Wall-clock benchmark across seq_len 512→8192. Spoiler: sparse is slower here (see Analysis). |
| `shakes.py` | Trains a tiny char-level GPT on TinyShakespeare per attention type. Reports train/val loss. |

### Run order

```
pip install -r requirements.txt
cd "Task 1"
python dense_attn.py && python sliding_window_attn.py && python global_window_attn.py
python sanity.py && python boom.py && python speedrun.py && python shakes.py
```

### What the benchmark actually shows

These implementations still compute the full N×N score matrix and mask afterward — the masked-out matmuls still happen, they're just zeroed. So sparse ends up *slower* than dense (extra mask-building overhead, same compute). A real sparse kernel would skip the masked blocks entirely. The benchmark is honest about this.

---

## Task 3 — Attention-Aware KV Cache Compression

Took Qwen2.5-0.5B apart to understand where attention mass actually goes, then built three cache eviction policies and ran them through perplexity + needle-in-haystack evals to see which one breaks least.

### Files

| File | What it does |
|------|-------------|
| `setup_model.py` | Loads Qwen2.5-0.5B with eager attention, confirms cache/attention access works. |
| `attention_viz.py` | Extracts attention weights across all 24 layers, produces the heatmap and sink-vs-rest plots. |
| `yeet.py` | The three eviction policies: sliding window, sink-aware (StreamingLLM-style), and H2O (heavy-hitter). |
| `loopy.py` | Generation loop wiring all three policies with the RoPE position patch. |
| `needle.py` | Perplexity + needle-in-haystack eval at a fixed budget. Uses a Jujutsu Kaisen wiki article as the haystack because why not. |
| `sweep.py` | Budget sweep across [32, 64, 128, 256, 512] — both metrics. Produces the analysis plots. |

### Run order

```
pip install -r requirements.txt
cd "Task 3"
python setup_model.py
python attention_viz.py
python needle.py
python sweep.py
```

### Key findings

**Sink structure is real and measurable.** Layers 0–2 spread attention uniformly. Layers 3–21 dump 25–50% of mass onto token 0 (a learned no-op target, not a semantically important token). Layers 22–23 collapse back to ~10%. Three-phase structure, not a single global pattern.

**Sink-aware eviction dominates.** On perplexity, it's the clear winner at every budget — ~200 ppl at cap=64 vs ~800 for sliding window and ~2300 for H2O. It's also the first to recover the needle (at cap=256).

**H2O has a recency bias.** Tokens that arrived earlier have had more steps to accumulate score, so H2O partially protects "old" tokens rather than "important" ones. This shows up in the sweep — it only catches up to sink-aware at very high budgets (512).

**RoPE after eviction is the trap.** Cached keys carry RoPE from their original positions. After eviction, if you don't patch `cache_position`, the new query gets RoPE applied at the wrong index and the model generates fluent garbage. Fix: override `cache_position` to equal the compressed cache size. This is approximate (cached keys keep their original phase) but it's what StreamingLLM does.

---

## Writeups

Handwritten notes and task writeups are in `Writeups/`. These cover the theory, observations, and honest limitations — including what I didn't build (per-layer budgets, full RoPE re-encoding) and why.

---

## Analysis

Both tasks have an `Analysis/` subfolder with the generated plots:

**Task 1:** `benchmark_plot.png` — wall-clock comparison across sequence lengths.

**Task 3:** `attn_heatmap.png` (attention mass per position per layer), `sink_vs_rest.png` (first-4-tokens vs rest per layer), `sweep_ppl.png` (perplexity vs cache budget), `sweep_needle.png` (needle recovery vs budget).

---

## Setup

```
pip install -r requirements.txt
```

Needs PyTorch ≥ 2.0, transformers ≥ 4.36 (for Qwen2 support). Task 3 downloads Qwen2.5-0.5B weights on first run (~1GB). `shakes.py` downloads TinyShakespeare (~1MB).

GPU recommended for Task 1 benchmarks (CPU works but memory tracking is unavailable) and Task 3 sweeps (slow on CPU).
