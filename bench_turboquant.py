"""Benchmark: FA4 baseline vs TurboQuant dequant on A100."""
import torch
from flash_attn.cute.benchmark import benchmark_forward
from flash_attn.cute import flash_attn_func

def run():
    device = "cuda"
    dtype = torch.bfloat16

    configs = [
        (2, 1024, 16, 128),
        (2, 2048, 16, 128),
        (2, 4096, 16, 128),
        (4, 2048, 32, 128),
    ]

    for batch, seqlen, nheads, headdim in configs:
        q = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
        k = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
        v = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)

        label = f"b={batch} s={seqlen} h={nheads} d={headdim}"
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"{'='*60}")

        # Baseline: full precision causal
        benchmark_forward(flash_attn_func, q, k, v, causal=True, desc=f"[BF16 causal] {label}")

        # Baseline: full precision non-causal
        benchmark_forward(flash_attn_func, q, k, v, causal=False, desc=f"[BF16 non-causal] {label}")

        # TurboQuant 4-bit causal
        benchmark_forward(flash_attn_func, q, k, v, causal=True, num_bits=4, desc=f"[TQ 4-bit causal] {label}")

        # TurboQuant 2-bit causal
        benchmark_forward(flash_attn_func, q, k, v, causal=True, num_bits=2, desc=f"[TQ 2-bit causal] {label}")

if __name__ == "__main__":
    run()
