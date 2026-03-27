
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
    packed_hd = headdim // 2

    packed_k = torch.randint(0, 256, (total_pages, page_size, nheads_kv, packed_hd),
                             dtype=torch.uint8, device=device)
    packed_v = torch.randint(0, 256, (total_pages, page_size, nheads_kv, packed_hd),
                             dtype=torch.uint8, device=device)
    page_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(batch, num_pages_per_seq)
    codebook = CODEBOOK_4BIT.to(dtype=dtype, device=device)
    return packed_k, packed_v, page_table, codebook


def bench(batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, page_size, dtype, device, repeats=30):
    from flash_attn.cute.dequant_kernel import dequant_paged_kv
    from flash_attn.cute import flash_attn_func

    packed_k, packed_v, page_table, codebook = make_inputs(
        batch, seqlen_k, nheads_kv, headdim, page_size, dtype, device)
    q = torch.randn(batch, seqlen_q, nheads, headdim, dtype=dtype, device=device)

    label = f"b={batch} sq={seqlen_q} sk={seqlen_k} h={nheads}/{nheads_kv} d={headdim}"

    try:
        k_fp = dequant_paged_kv(packed_k, page_table, codebook, page_size, headdim, nheads_kv, seqlen_k, batch)
        v_fp = dequant_paged_kv(packed_v, page_table, codebook, page_size, headdim, nheads_kv, seqlen_k, batch)
        flash_attn_func(q, k_fp, v_fp, causal=True)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[FAILED] {label}: {e}")
        traceback.print_exc()
        return

    def dequant_only():
        dequant_paged_kv(packed_k, page_table, codebook, page_size, headdim, nheads_kv, seqlen_k, batch)
        dequant_paged_kv(packed_v, page_table, codebook, page_size, headdim, nheads_kv, seqlen_k, batch)

    def dequant_plus_attn():
        k = dequant_paged_kv(packed_k, page_table, codebook, page_size, headdim, nheads_kv, seqlen_k, batch)
        v = dequant_paged_kv(packed_v, page_table, codebook, page_size, headdim, nheads_kv, seqlen_k, batch)
        flash_attn_func(q, k, v, causal=True)

    def baseline_attn():
        flash_attn_func(q, k_fp, v_fp, causal=True)

    t_dq = benchmark.Timer(stmt="fn()", globals={"fn": dequant_only}, num_threads=torch.get_num_threads())
    t_dqa = benchmark.Timer(stmt="fn()", globals={"fn": dequant_plus_attn}, num_threads=torch.get_num_threads())
    t_base = benchmark.Timer(stmt="fn()", globals={"fn": baseline_attn}, num_threads=torch.get_num_threads())

    m_dq = t_dq.timeit(repeats)
    m_base = t_base.timeit(repeats)
    m_dqa = t_dqa.timeit(repeats)

    print(f"  {label}")
    print(f"    Dequant K+V:      {m_dq.mean * 1e6:8.2f} us")
    print(f"    Baseline FA4:     {m_base.mean * 1e6:8.2f} us")
    print(f"    Dequant + FA4:    {m_dqa.mean * 1e6:8.2f} us")
    print(f"    Overhead:         {(m_dqa.mean - m_base.mean) * 1e6:8.2f} us ({m_dqa.mean / m_base.mean:.2f}x)")


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

    for batch, sq, sk, nh, nhkv, hd in configs:
        bench(batch, sq, sk, nh, nhkv, hd, page_size, dtype, device)


if __name__ == "__main__":
    main()
