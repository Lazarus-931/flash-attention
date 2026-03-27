# TurboQuant Dequant Benchmark Results

Config: bf16, GQA 32/8 heads, d=128, page_size=64, causal, seqlen_q=1 (decode)

## H100 80GB HBM3 (SM90)

### Fused CuTeDSL (paged KV path)

Baseline (fp16 paged KV, non-TMA cp.async):
  b=1 sk=2048: 46 us | b=1 sk=4096: 59 us | b=1 sk=8192: 111 us | b=1 sk=16384: 220 us

v1 — Byte-by-byte serial load, register codebook:
  b=1 sk=2048: 3823 us (81x) | b=1 sk=4096: 7691 us (130x) | b=1 sk=8192: 15559 us (140x) | b=1 sk=16384: 31398 us (143x)

v2 — Int32 vectorized loads, constexpr unrolled:
  b=1 sk=2048: 3205 us (68x) | b=1 sk=4096: 6411 us (109x) | b=1 sk=8192: 12868 us (116x) | b=1 sk=16384: 25997 us (118x)

v3 — Fully unrolled cb[code], codebook hoisted:
  b=1 sk=2048: 570 us (12x) | b=1 sk=4096: 1138 us (19x) | b=1 sk=8192: 2282 us (21x) | b=1 sk=16384: 4807 us (22x)

v4 — Polynomial approximation (REVERTED, slower than v3):
  b=1 sk=2048: 2338 us (50x) | b=1 sk=4096: 4755 us (81x) | b=1 sk=8192: 9688 us (87x) | b=1 sk=16384: 19349 us (88x)

v5 — Parallelized across thread partitions:
  b=1 sk=2048: 122 us (2.7x) | b=1 sk=4096: 240 us (4.1x) | b=1 sk=8192: 476 us (4.3x) | b=1 sk=16384: 1135 us (5.2x)

v6 — Shared memory codebook (FAILED, CuTeDSL limitation)

### Separate CUDA C kernel (non-paged FA4)

v7 — CUDA C dequant + standard flash_attn_func:
  b=1 sk=2048: dequant 29 us + FA4 52 us = 75 us (1.43x)
  b=1 sk=4096: dequant 53 us + FA4 54 us = 105 us (1.94x)
  b=1 sk=8192: dequant 100 us + FA4 100 us = 201 us (2.02x)
  b=1 sk=16384: dequant 198 us + FA4 191 us = 387 us (2.02x)
  b=8 sk=4096: dequant 389 us + FA4 56 us = 448 us (7.94x)
  b=16 sk=4096: dequant 769 us + FA4 92 us = 868 us (9.46x)

## A100 80GB (SM80)

Pending.

## H200 (SM90)

Pending.

## B200 (SM100)

Pending.
