"""FA4 paged KV benchmark: baseline vs TurboQuant 4-bit on H100.

Uses page_size=64 (different from tile_n=128) to force the non-TMA
paged KV path in the SM90 kernel, which is where TurboQuant dequant lives.
"""
import torch
import torch.utils.benchmark as benchmark
import traceback


def make_paged_kv_inputs(batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, page_size, dtype, device):
    num_pages_per_seq = (seqlen_k + page_size - 1) // page_size
    total_pages = batch * num_pages_per_seq

    q = torch.randn(batch * seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(total_pages, page_size, nheads_kv, headdim, dtype=dtype, device=device)
    v = torch.randn(total_pages, page_size, nheads_kv, headdim, dtype=dtype, device=device)

    page_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(batch, num_pages_per_seq)
    cu_seqlens_q = torch.arange(0, batch * seqlen_q + 1, seqlen_q, dtype=torch.int32, device=device)
    seqused_k = torch.full((batch,), seqlen_k, dtype=torch.int32, device=device)

    return q, k, v, cu_seqlens_q, seqused_k, page_table


def bench_config(batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, page_size, dtype, device, num_bits, repeats=30):
    from flash_attn.cute import flash_attn_varlen_func

    q, k, v, cu_seqlens_q, seqused_k, page_table = make_paged_kv_inputs(
        batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, page_size, dtype, device
    )

    label = f"b={batch} sq={seqlen_q} sk={seqlen_k} h={nheads}/{nheads_kv} d={headdim} pg={page_size} bits={num_bits}"

    # Warmup (includes JIT compilation)
    try:
        for _ in range(3):
            flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=seqlen_q,
                seqused_k=seqused_k,
                page_table=page_table,
                causal=True,
                num_bits=num_bits,
            )
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[FAILED] {label}")
        print(f"  Error: {e}")
        traceback.print_exc()
        return None

    t = benchmark.Timer(
        stmt="fn(q, k, v, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_sq, seqused_k=seqused_k, page_table=page_table, causal=True, num_bits=num_bits)",
        globals={
            "fn": flash_attn_varlen_func,
            "q": q, "k": k, "v": v,
            "cu_seqlens_q": cu_seqlens_q,
            "max_sq": seqlen_q,
            "seqused_k": seqused_k,
            "page_table": page_table,
            "num_bits": num_bits,
        },
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    print(f"[FA4 paged KV] {label}")
    print(f"  {m.mean * 1e6:.2f} us (mean of {repeats} runs)")
    return m


def main():
    device = "cuda"
    dtype = torch.bfloat16
    # page_size=64 forces non-TMA paged KV path on SM90 (tile_n=128)
    page_size = 64

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"dtype: {dtype}, page_size: {page_size}")
    print("=" * 70)

    configs = [
        # (batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim)
        # Decode-like
        (1,  1, 2048,  32, 8, 128),
        (1,  1, 4096,  32, 8, 128),
        (1,  1, 8192,  32, 8, 128),
        (1,  1, 16384, 32, 8, 128),
        (8,  1, 4096,  32, 8, 128),
        (16, 1, 4096,  32, 8, 128),
    ]

    for bits in [32, 4]:
        tag = "BASELINE (fp16)" if bits == 32 else f"TURBOQUANT ({bits}-bit)"
        print(f"\n{'='*70}")
        print(f"  {tag}")
        print(f"{'='*70}")
        for batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim in configs:
            bench_config(batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim,
                         page_size, dtype, device, num_bits=bits)


if __name__ == "__main__":
    main()
