
import torch
import torch.utils.benchmark as benchmark
import traceback

CODEBOOK_4BIT = torch.tensor([
    -2.73259458, -2.06902289, -1.61805216, -1.25623664,
    -0.94234516, -0.65676275, -0.38805059, -0.12839581,
     0.12839581,  0.38805059,  0.65676275,  0.94234516,
     1.25623664,  1.61805216,  2.06902289,  2.73259458,
])


def make_inputs(batch, seqlen_k, nheads_kv, headdim, page_size, dtype, device):
    num_pages_per_seq = (seqlen_k + page_size - 1) // page_size
    total_pages = batch * num_pages_per_seq
    packed_head_dim = headdim // 2

    packed_kv = torch.randint(0, 256, (total_pages, page_size, nheads_kv, packed_head_dim),
                              dtype=torch.uint8, device=device)
    page_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(batch, num_pages_per_seq)
    codebook = CODEBOOK_4BIT.to(dtype=dtype, device=device)

    return packed_kv, page_table, codebook


def bench_dequant_only(batch, seqlen_k, nheads, nheads_kv, headdim, page_size, dtype, device, repeats=30):
    from flash_attn.cute.dequant_kernel import dequant_paged_kv

    packed_kv, page_table, codebook = make_inputs(batch, seqlen_k, nheads_kv, headdim, page_size, dtype, device)

    try:
        out = dequant_paged_kv(packed_kv, page_table, codebook, page_size, headdim, nheads_kv, seqlen_k, batch)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[FAILED] dequant: {e}")
        traceback.print_exc()
        return None

    label = f"b={batch} sk={seqlen_k} h={nheads}/{nheads_kv} d={headdim}"

    t = benchmark.Timer(
        stmt="fn(packed_kv, page_table, codebook, page_size, headdim, nheads_kv, seqlen_k, batch)",
        globals={
            "fn": dequant_paged_kv,
            "packed_kv": packed_kv, "page_table": page_table, "codebook": codebook,
            "page_size": page_size, "headdim": headdim, "nheads_kv": nheads_kv,
            "seqlen_k": seqlen_k, "batch": batch,
        },
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    print(f"[Dequant kernel] {label}: {m.mean * 1e6:.2f} us")
    return m


def bench_dequant_plus_attn(batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, page_size, dtype, device, repeats=30):
    from flash_attn.cute.dequant_kernel import dequant_paged_kv
    from flash_attn.cute import flash_attn_func

    packed_kv, page_table, codebook = make_inputs(batch, seqlen_k, nheads_kv, headdim, page_size, dtype, device)
    q = torch.randn(batch, seqlen_q, nheads, headdim, dtype=dtype, device=device)

    def dequant_then_attn():
        kv = dequant_paged_kv(packed_kv, page_table, codebook, page_size, headdim, nheads_kv, seqlen_k, batch)
        k = kv[:, :, :, :headdim]
        v = kv[:, :, :, :headdim]
        flash_attn_func(q, k, v, causal=True)

    try:
        dequant_then_attn()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[FAILED] dequant+attn: {e}")
        traceback.print_exc()
        return None

    label = f"b={batch} sq={seqlen_q} sk={seqlen_k} h={nheads}/{nheads_kv} d={headdim}"

    t = benchmark.Timer(
        stmt="fn()",
        globals={"fn": dequant_then_attn},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    print(f"[Dequant+Attn] {label}: {m.mean * 1e6:.2f} us")
    return m


def bench_baseline_attn(batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, dtype, device, repeats=30):
    from flash_attn.cute import flash_attn_func

    q = torch.randn(batch, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen_k, nheads_kv, headdim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen_k, nheads_kv, headdim, dtype=dtype, device=device)

    try:
        flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[FAILED] baseline: {e}")
        traceback.print_exc()
        return None

    label = f"b={batch} sq={seqlen_q} sk={seqlen_k} h={nheads}/{nheads_kv} d={headdim}"

    t = benchmark.Timer(
        stmt="fn(q, k, v, causal=True)",
        globals={"fn": flash_attn_func, "q": q, "k": k, "v": v},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    print(f"[Baseline FA4] {label}: {m.mean * 1e6:.2f} us")
    return m


def main():
    device = "cuda"
    dtype = torch.bfloat16
    page_size = 64

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"dtype: {dtype}, page_size: {page_size}")
    print("=" * 70)

    configs = [
        (1,  1, 2048,  32, 8, 128),
        (1,  1, 4096,  32, 8, 128),
        (1,  1, 8192,  32, 8, 128),
        (1,  1, 16384, 32, 8, 128),
        (8,  1, 4096,  32, 8, 128),
        (16, 1, 4096,  32, 8, 128),
    ]

    print("\n--- Dequant kernel only ---")
    for batch, sq, sk, nh, nhkv, hd in configs:
        bench_dequant_only(batch, sk, nh, nhkv, hd, page_size, dtype, device)

    print("\n--- Baseline FA4 (fp16, non-paged) ---")
    for batch, sq, sk, nh, nhkv, hd in configs:
        bench_baseline_attn(batch, sq, sk, nh, nhkv, hd, dtype, device)

    print("\n--- Dequant + FA4 (separate kernel) ---")
    for batch, sq, sk, nh, nhkv, hd in configs:
        bench_dequant_plus_attn(batch, sq, sk, nh, nhkv, hd, page_size, dtype, device)


if __name__ == "__main__":
    main()
