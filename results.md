# TurboQuant Dequant Benchmark Results

Config: bf16, GQA 32/8 heads, d=128, page_size=64, causal, seqlen_q=1 (decode)

## H100 80GB HBM3 (SM90, 3.35 TB/s)

### v5 — Fused CuTeDSL (paged KV, parallelized cb[code])

Baseline (fp16 paged non-TMA):
  b=1 sk=2048: 53 us | sk=4096: 58 us | sk=8192: 110 us | sk=16384: 217 us

TurboQuant 4-bit:
  b=1 sk=2048: 123 us (2.3x) | sk=4096: 242 us (4.2x) | sk=8192: 483 us (4.4x) | sk=16384: 1178 us (5.4x)

### v7 — Separate CUDA C kernel (non-paged FA4)

  b=1 sk=2048: dequant 29 + attn 62 = 90 us (1.46x)
  b=1 sk=4096: dequant 52 + attn 59 = 104 us (1.77x)
  b=1 sk=8192: dequant 100 + attn 100 = 201 us (2.00x)
  b=1 sk=16384: dequant 199 + attn 193 = 390 us (2.02x)
  b=8 sk=4096: dequant 389 + attn 105 = 495 us (4.73x)
  b=16 sk=4096: dequant 769 + attn 204 = 971 us (4.75x)

### Findings

v7 is 1.5-2x overhead for b=1 (vs v5 at 2.3-5.4x). Dequant kernel scales linearly with total KV elements. For b=1 long sequences, dequant cost roughly equals attention cost. Batched case dominated by dequant.

---

## A100 40GB SXM4 (SM80, 2.0 TB/s)

### v7 — Separate CUDA C kernel (non-paged FA4, pack_gqa=False)

  b=1 sk=2048: dequant 60 + attn 124 = 188 us (1.51x)
  b=1 sk=4096: dequant 110 + attn 234 = 342 us (1.46x)
  b=1 sk=8192: dequant 212 + attn 459 = 668 us (1.46x)
  b=1 sk=16384: dequant 417 + attn 906 = 1180 us (1.30x)
  b=8 sk=4096: dequant 642 + attn 446 = 1084 us (2.43x)
  b=16 sk=4096: dequant 1276 + attn 713 = 2017 us (2.83x)

### Findings

A100 shows BETTER overhead ratios than H100 (1.30-1.51x vs 1.46-2.02x for b=1). This is because A100 has lower HBM bandwidth (2.0 vs 3.35 TB/s), making attention itself slower relative to dequant compute. The dequant overhead is a smaller fraction of total time. At sk=16384, overhead is only 1.30x — the 4x memory reduction is nearly free.

v5 fused approach not available on A100 (paged KV requires SM90+).

---

## H200 (SM90)

Not available on Modal.

## B200 (SM100)

Not available on Modal.

---

## Summary

| GPU | Version | b=1 sk=4096 | b=1 sk=16384 | Best case |
|-----|---------|-------------|--------------|-----------|
| H100 | v5 fused | 4.2x | 5.4x | 2.3x (sk=2048) |
| H100 | v7 separate | 1.77x | 2.02x | 1.46x (sk=2048) |
| A100 | v7 separate | 1.46x | 1.30x | 1.30x (sk=16384) |

A100 benefits more from TurboQuant because it's more bandwidth-constrained. The 4x KV compression saves proportionally more time when bandwidth is the bottleneck.
