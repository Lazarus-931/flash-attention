
import torch
from torch.utils.cpp_extension import load_inline

DEQUANT_CUDA = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename scalar_t, int NUM_BITS>
__global__ void dequant_paged_kv_kernel(
    const uint8_t* __restrict__ packed_kv,
    scalar_t* __restrict__ out_kv,
    const int32_t* __restrict__ page_table,
    const scalar_t* __restrict__ codebook,
    int page_size,
    int head_dim,
    int num_heads_kv,
    int seqlen_k,
    int batch_size
) {
    constexpr int ELEMS_PER_BYTE = 8 / NUM_BITS;
    constexpr int MASK = (1 << NUM_BITS) - 1;
    int packed_head_dim = head_dim / ELEMS_PER_BYTE;

    __shared__ scalar_t cb[1 << NUM_BITS];
    if (threadIdx.x < (1 << NUM_BITS))
        cb[threadIdx.x] = codebook[threadIdx.x];
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = batch_size * seqlen_k * num_heads_kv * head_dim;
    if (tid >= total_elems) return;

    int d = tid % head_dim;
    int h = (tid / head_dim) % num_heads_kv;
    int s = (tid / (head_dim * num_heads_kv)) % seqlen_k;
    int b = tid / (head_dim * num_heads_kv * seqlen_k);

    int page_idx = page_table[b * ((seqlen_k + page_size - 1) / page_size) + s / page_size];
    int page_offset = s % page_size;

    int packed_byte_idx = d / ELEMS_PER_BYTE;
    int sub_idx = d % ELEMS_PER_BYTE;

    int packed_offset = ((page_idx * page_size + page_offset) * num_heads_kv + h) * packed_head_dim + packed_byte_idx;
    uint8_t packed_val = packed_kv[packed_offset];
    int code = (packed_val >> (sub_idx * NUM_BITS)) & MASK;

    out_kv[tid] = cb[code];
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
    auto options = torch::TensorOptions().dtype(codebook.dtype()).device(packed_kv.device());
    auto out = torch::empty({batch_size, seqlen_k, num_heads_kv, head_dim}, options);

    int total = batch_size * seqlen_k * num_heads_kv * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_SWITCH(codebook.scalar_type(), "dequant_paged_kv",
        AT_DISPATCH_CASE(at::ScalarType::Half,
            [&] { dequant_paged_kv_kernel<scalar_t, 4><<<blocks, threads>>>(
                packed_kv.data_ptr<uint8_t>(), out.data_ptr<scalar_t>(),
                page_table.data_ptr<int32_t>(), codebook.data_ptr<scalar_t>(),
                page_size, head_dim, num_heads_kv, seqlen_k, batch_size); })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16,
            [&] { dequant_paged_kv_kernel<scalar_t, 4><<<blocks, threads>>>(
                packed_kv.data_ptr<uint8_t>(), out.data_ptr<scalar_t>(),
                page_table.data_ptr<int32_t>(), codebook.data_ptr<scalar_t>(),
                page_size, head_dim, num_heads_kv, seqlen_k, batch_size); })
    );
    return out;
}
"""

DEQUANT_CPP = """
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
);
"""

_module = None

def get_dequant_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="turboquant_dequant",
            cpp_sources=[DEQUANT_CPP],
            cuda_sources=[DEQUANT_CUDA],
            functions=["dequant_paged_kv"],
            verbose=False,
        )
    return _module


def dequant_paged_kv(packed_kv, page_table, codebook, page_size, head_dim,
                     num_heads_kv, seqlen_k, batch_size, num_bits=4):
    mod = get_dequant_module()
    return mod.dequant_paged_kv(
        packed_kv, page_table, codebook,
        page_size, head_dim, num_heads_kv, seqlen_k, batch_size, num_bits
    )
