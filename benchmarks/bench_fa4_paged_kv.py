"""FA4 paged KV baseline benchmark on A100.

Measures forward pass latency for paged KV attention using flash_attn_varlen_func.
Covers decode-like (seqlen_q=1) and prefill-like configs.
"""
import torch
import torch.utils.benchmark as benchmark


def make_paged_kv_inputs(batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, page_size, dtype, device):
    """Create paged KV inputs for flash_attn_varlen_func."""
    num_pages_per_seq = (seqlen_k + page_size - 1) // page_size
    total_pages = batch * num_pages_per_seq

    q = torch.randn(batch * seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(total_pages, page_size, nheads_kv, headdim, dtype=dtype, device=device)
    v = torch.randn(total_pages, page_size, nheads_kv, headdim, dtype=dtype, device=device)

    # page_table: (batch, num_pages_per_seq), identity mapping
    page_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(batch, num_pages_per_seq)

    # cu_seqlens for varlen
    cu_seqlens_q = torch.arange(0, batch * seqlen_q + 1, seqlen_q, dtype=torch.int32, device=device)
    seqused_k = torch.full((batch,), seqlen_k, dtype=torch.int32, device=device)

    return q, k, v, cu_seqlens_q, seqused_k, page_table


def bench_config(batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, page_size, dtype, device, repeats=30):
    from flash_attn.cute import flash_attn_varlen_func

    q, k, v, cu_seqlens_q, seqused_k, page_table = make_paged_kv_inputs(
        batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, page_size, dtype, device
    )

    # Warmup
    for _ in range(3):
        flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=seqlen_q,
            seqused_k=seqused_k,
            page_table=page_table,
            causal=True,
        )
    torch.cuda.synchronize()

    label = f"b={batch} sq={seqlen_q} sk={seqlen_k} h={nheads}/{nheads_kv} d={headdim} pg={page_size}"

    t = benchmark.Timer(
        stmt="fn(q, k, v, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_sq, seqused_k=seqused_k, page_table=page_table, causal=True)",
        globals={
            "fn": flash_attn_varlen_func,
            "q": q, "k": k, "v": v,
            "cu_seqlens_q": cu_seqlens_q,
            "max_sq": seqlen_q,
            "seqused_k": seqused_k,
            "page_table": page_table,
        },
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    print(f"[FA4 paged KV] {label}")
    print(f"  {m}")
    return m


def main():
    device = "cuda"
    dtype = torch.bfloat16
    page_size = 128

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"dtype: {dtype}, page_size: {page_size}")
    print("=" * 70)

    # Decode-like configs (seqlen_q=1, long KV cache)
    print("\n--- Decode-like (seqlen_q=1) ---")
    for batch, seqlen_k, nheads, nheads_kv, headdim in [
        (1,  2048,  32, 8, 128),
        (1,  4096,  32, 8, 128),
        (1,  8192,  32, 8, 128),
        (1,  16384, 32, 8, 128),
        (8,  4096,  32, 8, 128),
        (16, 4096,  32, 8, 128),
    ]:
        bench_config(batch, 1, seqlen_k, nheads, nheads_kv, headdim, page_size, dtype, device)

    # Prefill-like configs
    print("\n--- Prefill-like ---")
    for batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim in [
        (2, 512,  512,  32, 8, 128),
        (2, 1024, 1024, 32, 8, 128),
        (2, 2048, 2048, 32, 8, 128),
    ]:
        bench_config(batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, page_size, dtype, device)


if __name__ == "__main__":
    main()
