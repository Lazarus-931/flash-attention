# TurboQuant Dequant Benchmark Results

Config: bf16, GQA 32/8 heads, d=128, page_size=64, causal, seqlen_q=1 (decode)

## H100 80GB HBM3 (SM90, 3.35 TB/s)

### v5 — Fused CuTeDSL (paged KV, parallelized cb[code])

Baseline (fp16 paged non-TMA):
  b=1 sk=2048: 53 us | sk=4096: 58 us | sk=8192: 110 us | sk=16384: 217 us

TurboQuant 4-bit:
  b=1 sk=2048: 123 us (2.3x) | sk=4096: 242 us (4.2x) | sk=8192: 483 us (4.4x) | sk=16384: 1178 us (5.4x)

### v7 — Separate CUDA C kernel, 2x launch (non-paged FA4)

  b=1 sk=2048: dequant 29 + attn 62 = 90 us (1.46x)
  b=1 sk=4096: dequant 52 + attn 59 = 104 us (1.77x)
  b=1 sk=8192: dequant 100 + attn 100 = 201 us (2.00x)
  b=1 sk=16384: dequant 199 + attn 193 = 390 us (2.02x)

### v8 — Fused K+V single kernel launch (non-paged FA4)

  b=1 sk=2048: dequant 15 + attn 48 = 61 us (1.27x)
  b=1 sk=4096: dequant 28 + attn 53 = 81 us (1.51x)
  b=1 sk=8192: dequant 55 + attn 101 = 154 us (1.54x)
  b=1 sk=16384: dequant 107 + attn 194 = 295 us (1.52x)
  b=8 sk=4096: dequant 206 + attn 104 = 312 us (2.99x)
  b=16 sk=4096: dequant 408 + attn 202 = 613 us (3.03x)

Fused K+V dequant is 0.54x of separate (nearly 2x faster). Each thread loads and dequants both K and V with one page_table lookup, one smem codebook load, shared index computation.

---

## A100 40GB SXM4 (SM80, 2.0 TB/s)

### v7 — Separate CUDA C kernel, 2x launch

  b=1 sk=2048: dequant 60 + attn 124 = 188 us (1.51x)
  b=1 sk=4096: dequant 110 + attn 234 = 342 us (1.46x)
  b=1 sk=8192: dequant 212 + attn 459 = 668 us (1.46x)
  b=1 sk=16384: dequant 417 + attn 906 = 1180 us (1.30x)

### v8 — Fused K+V single kernel launch

  b=1 sk=2048: dequant 33 + attn 120 = 152 us (1.26x)
  b=1 sk=4096: dequant 60 + attn 232 = 291 us (1.26x)
  b=1 sk=8192: dequant 118 + attn 454 = 568 us (1.25x)
  b=1 sk=16384: dequant 229 + attn 899 = 1123 us (1.25x)
  b=8 sk=4096: dequant 418 + attn 529 = 944 us (1.79x)
  b=16 sk=4096: dequant 793 + attn 806 = 1595 us (1.98x)

A100 benefits most: 1.25x overhead across all b=1 sequence lengths. The 4x memory reduction costs only 25% more latency.

---

## Summary

| GPU | Version | b=1 sk=2048 | b=1 sk=8192 | b=1 sk=16384 |
|-----|---------|-------------|-------------|--------------|
| H100 | v5 fused CuTeDSL | 2.3x | 4.4x | 5.4x |
| H100 | v7 separate 2x | 1.46x | 2.00x | 2.02x |
| H100 | v8 fused K+V | 1.27x | 1.54x | 1.52x |
| A100 | v7 separate 2x | 1.51x | 1.46x | 1.30x |
| A100 | v8 fused K+V | 1.26x | 1.25x | 1.25x |

Fusing K+V into one launch cut dequant time in half. A100 consistently at 1.25x overhead (b=1). H100 at 1.27-1.54x. Bandwidth-constrained GPUs benefit more.
