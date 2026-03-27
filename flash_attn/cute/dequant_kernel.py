
import torch
from torch.utils.cpp_extension import load_inline

DEQUANT_CUDA = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename scalar_t, int NUM_BITS>
__global__ void dequant_paged_kv_kernel(
    const uint8_t* __restrict__ packed,
    scalar_t* __restrict__ out,
    const int32_t* __restrict__ page_table,
    const scalar_t* __restrict__ codebook,
    int page_size,
    int head_dim,
    int num_heads_kv,
    int seqlen_k,
    int max_pages_per_seq
) {
    constexpr int ELEMS_PER_BYTE = 8 / NUM_BITS;
    constexpr int MASK = (1 << NUM_BITS) - 1;
    const int packed_head_dim = head_dim / ELEMS_PER_BYTE;

    __shared__ scalar_t cb[1 << NUM_BITS];
    if (threadIdx.x < (1 << NUM_BITS))
        cb[threadIdx.x] = codebook[threadIdx.x];
    __syncthreads();

    const int batch_idx = blockIdx.y;
    const int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_per_batch = seqlen_k * num_heads_kv * head_dim;
    if (elem_idx >= total_per_batch) return;

    const int d = elem_idx % head_dim;
    const int h = (elem_idx / head_dim) % num_heads_kv;
    const int s = elem_idx / (head_dim * num_heads_kv);

    const int page_idx = page_table[batch_idx * max_pages_per_seq + s / page_size];
    const int page_offset = s % page_size;
    const int byte_idx = d / ELEMS_PER_BYTE;
    const int sub_idx = d % ELEMS_PER_BYTE;

    const int packed_offset = ((page_idx * page_size + page_offset) * num_heads_kv + h) * packed_head_dim + byte_idx;
    const int code = (packed[packed_offset] >> (sub_idx * NUM_BITS)) & MASK;

    const int out_offset = ((batch_idx * seqlen_k + s) * num_heads_kv + h) * head_dim + d;
    out[out_offset] = cb[code];
}

template <typename scalar_t, int NUM_BITS>
__global__ void dequant_paged_kv_fused_kernel(
    const uint8_t* __restrict__ packed_k,
    const uint8_t* __restrict__ packed_v,
    scalar_t* __restrict__ out_k,
    scalar_t* __restrict__ out_v,
    const int32_t* __restrict__ page_table,
    const scalar_t* __restrict__ codebook,
    int page_size,
    int head_dim,
    int num_heads_kv,
    int seqlen_k,
    int max_pages_per_seq
) {
    constexpr int ELEMS_PER_BYTE = 8 / NUM_BITS;
    constexpr int MASK = (1 << NUM_BITS) - 1;
    const int packed_head_dim = head_dim / ELEMS_PER_BYTE;

    __shared__ scalar_t cb[1 << NUM_BITS];
    if (threadIdx.x < (1 << NUM_BITS))
        cb[threadIdx.x] = codebook[threadIdx.x];
    __syncthreads();

    const int batch_idx = blockIdx.y;
    const int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_per_batch = seqlen_k * num_heads_kv * head_dim;
    if (elem_idx >= total_per_batch) return;

    const int d = elem_idx % head_dim;
    const int h = (elem_idx / head_dim) % num_heads_kv;
    const int s = elem_idx / (head_dim * num_heads_kv);

    const int page_idx = page_table[batch_idx * max_pages_per_seq + s / page_size];
    const int page_offset = s % page_size;
    const int byte_idx = d / ELEMS_PER_BYTE;
    const int sub_idx = d % ELEMS_PER_BYTE;

    const int packed_offset = ((page_idx * page_size + page_offset) * num_heads_kv + h) * packed_head_dim + byte_idx;
    const int out_offset = ((batch_idx * seqlen_k + s) * num_heads_kv + h) * head_dim + d;

    const int code_k = (packed_k[packed_offset] >> (sub_idx * NUM_BITS)) & MASK;
    const int code_v = (packed_v[packed_offset] >> (sub_idx * NUM_BITS)) & MASK;
    out_k[out_offset] = cb[code_k];
    out_v[out_offset] = cb[code_v];
}

template <typename scalar_t, int NUM_BITS>
void launch_dequant(
    const uint8_t* packed, scalar_t* out, const int32_t* page_table,
    const scalar_t* codebook, int page_size, int head_dim,
    int num_heads_kv, int seqlen_k, int batch_size, int max_pages_per_seq
) {
    const int total_per_batch = seqlen_k * num_heads_kv * head_dim;
    const int threads = 256;
    const int blocks_x = (total_per_batch + threads - 1) / threads;
    dim3 grid(blocks_x, batch_size);
    dequant_paged_kv_kernel<scalar_t, NUM_BITS><<<grid, threads>>>(
        packed, out, page_table, codebook,
        page_size, head_dim, num_heads_kv, seqlen_k, max_pages_per_seq
    );
}

template <typename scalar_t, int NUM_BITS>
void launch_dequant_fused(
    const uint8_t* packed_k, const uint8_t* packed_v,
    scalar_t* out_k, scalar_t* out_v, const int32_t* page_table,
    const scalar_t* codebook, int page_size, int head_dim,
    int num_heads_kv, int seqlen_k, int batch_size, int max_pages_per_seq
) {
    const int total_per_batch = seqlen_k * num_heads_kv * head_dim;
    const int threads = 256;
    const int blocks_x = (total_per_batch + threads - 1) / threads;
    dim3 grid(blocks_x, batch_size);
    dequant_paged_kv_fused_kernel<scalar_t, NUM_BITS><<<grid, threads>>>(
        packed_k, packed_v, out_k, out_v, page_table, codebook,
        page_size, head_dim, num_heads_kv, seqlen_k, max_pages_per_seq
    );
}

torch::Tensor dequant_paged_kv(
    torch::Tensor packed_kv,
    torch::Tensor page_table,
    torch::Tensor codebook,
    int page_size,
    int head_dim,
    int num_heads_kv,
    int seqlen_k,
    int batch_size,
    int num_bits
) {
    auto opts = torch::TensorOptions().dtype(codebook.dtype()).device(packed_kv.device());
    auto out = torch::empty({batch_size, seqlen_k, num_heads_kv, head_dim}, opts);
    int max_pages_per_seq = page_table.size(1);

    AT_DISPATCH_SWITCH(codebook.scalar_type(), "dequant_paged_kv",
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            if (num_bits == 2) launch_dequant<scalar_t, 2>(
                packed_kv.data_ptr<uint8_t>(), out.data_ptr<scalar_t>(),
                page_table.data_ptr<int32_t>(), codebook.data_ptr<scalar_t>(),
                page_size, head_dim, num_heads_kv, seqlen_k, batch_size, max_pages_per_seq);
            else if (num_bits == 4) launch_dequant<scalar_t, 4>(
                packed_kv.data_ptr<uint8_t>(), out.data_ptr<scalar_t>(),
                page_table.data_ptr<int32_t>(), codebook.data_ptr<scalar_t>(),
                page_size, head_dim, num_heads_kv, seqlen_k, batch_size, max_pages_per_seq);
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            if (num_bits == 2) launch_dequant<scalar_t, 2>(
                packed_kv.data_ptr<uint8_t>(), out.data_ptr<scalar_t>(),
                page_table.data_ptr<int32_t>(), codebook.data_ptr<scalar_t>(),
                page_size, head_dim, num_heads_kv, seqlen_k, batch_size, max_pages_per_seq);
            else if (num_bits == 4) launch_dequant<scalar_t, 4>(
                packed_kv.data_ptr<uint8_t>(), out.data_ptr<scalar_t>(),
                page_table.data_ptr<int32_t>(), codebook.data_ptr<scalar_t>(),
                page_size, head_dim, num_heads_kv, seqlen_k, batch_size, max_pages_per_seq);
        })
    );
    return out;
}

std::vector<torch::Tensor> dequant_paged_kv_fused(
    torch::Tensor packed_k,
    torch::Tensor packed_v,
    torch::Tensor page_table,
    torch::Tensor codebook,
    int page_size,
    int head_dim,
    int num_heads_kv,
    int seqlen_k,
    int batch_size,
    int num_bits
) {
    auto opts = torch::TensorOptions().dtype(codebook.dtype()).device(packed_k.device());
    auto out_k = torch::empty({batch_size, seqlen_k, num_heads_kv, head_dim}, opts);
    auto out_v = torch::empty({batch_size, seqlen_k, num_heads_kv, head_dim}, opts);
    int max_pages_per_seq = page_table.size(1);

    AT_DISPATCH_SWITCH(codebook.scalar_type(), "dequant_paged_kv_fused",
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            if (num_bits == 2) launch_dequant_fused<scalar_t, 2>(
                packed_k.data_ptr<uint8_t>(), packed_v.data_ptr<uint8_t>(),
                out_k.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>(),
                page_table.data_ptr<int32_t>(), codebook.data_ptr<scalar_t>(),
                page_size, head_dim, num_heads_kv, seqlen_k, batch_size, max_pages_per_seq);
            else if (num_bits == 4) launch_dequant_fused<scalar_t, 4>(
                packed_k.data_ptr<uint8_t>(), packed_v.data_ptr<uint8_t>(),
                out_k.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>(),
                page_table.data_ptr<int32_t>(), codebook.data_ptr<scalar_t>(),
                page_size, head_dim, num_heads_kv, seqlen_k, batch_size, max_pages_per_seq);
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            if (num_bits == 2) launch_dequant_fused<scalar_t, 2>(
                packed_k.data_ptr<uint8_t>(), packed_v.data_ptr<uint8_t>(),
                out_k.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>(),
                page_table.data_ptr<int32_t>(), codebook.data_ptr<scalar_t>(),
                page_size, head_dim, num_heads_kv, seqlen_k, batch_size, max_pages_per_seq);
            else if (num_bits == 4) launch_dequant_fused<scalar_t, 4>(
                packed_k.data_ptr<uint8_t>(), packed_v.data_ptr<uint8_t>(),
                out_k.data_ptr<scalar_t>(), out_v.data_ptr<scalar_t>(),
                page_table.data_ptr<int32_t>(), codebook.data_ptr<scalar_t>(),
                page_size, head_dim, num_heads_kv, seqlen_k, batch_size, max_pages_per_seq);
        })
    );
    return {out_k, out_v};
}
"""

DEQUANT_CPP = """
torch::Tensor dequant_paged_kv(
    torch::Tensor packed_kv, torch::Tensor page_table, torch::Tensor codebook,
    int page_size, int head_dim, int num_heads_kv, int seqlen_k,
    int batch_size, int num_bits);
std::vector<torch::Tensor> dequant_paged_kv_fused(
    torch::Tensor packed_k, torch::Tensor packed_v, torch::Tensor page_table,
    torch::Tensor codebook, int page_size, int head_dim, int num_heads_kv,
    int seqlen_k, int batch_size, int num_bits);
"""

_module = None

def get_dequant_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="turboquant_dequant",
            cpp_sources=[DEQUANT_CPP],
            cuda_sources=[DEQUANT_CUDA],
            functions=["dequant_paged_kv", "dequant_paged_kv_fused"],
            verbose=False,
        )
    return _module


def dequant_paged_kv(packed_kv, page_table, codebook, page_size, head_dim,
                     num_heads_kv, seqlen_k, batch_size, num_bits=4):
    return get_dequant_module().dequant_paged_kv(
        packed_kv, page_table, codebook,
        page_size, head_dim, num_heads_kv, seqlen_k, batch_size, num_bits)


def dequant_paged_kv_fused(packed_k, packed_v, page_table, codebook, page_size,
                           head_dim, num_heads_kv, seqlen_k, batch_size, num_bits=4):
    return get_dequant_module().dequant_paged_kv_fused(
        packed_k, packed_v, page_table, codebook,
        page_size, head_dim, num_heads_kv, seqlen_k, batch_size, num_bits)
