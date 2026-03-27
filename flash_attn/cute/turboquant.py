
import cutlass  # type: ignore
import cutlass.cute as cute  # type: ignore
from cutlass import Int32, Float32  # type: ignore

# Lloyd-Max centroids for the standard normal distribution

CODEBOOK = {
    1: [-0.79788456, 0.79788456],
    2: [-1.51041761, -0.45278003, 0.45278003, 1.51041761],
    3: [-2.15194570, -1.34390928, -0.75600528, -0.24509418,
         0.24509418,  0.75600528,  1.34390928,  2.15194570],
    4: [-2.73259458, -2.06902289, -1.61805216, -1.25623664,
        -0.94234516, -0.65676275, -0.38805059, -0.12839581,
         0.12839581,  0.38805059,  0.65676275,  0.94234516,
         1.25623664,  1.61805216,  2.06902289,  2.73259458],
}


class TurboQuant:
    def __init__(self, num_bits: int, dtype):
        assert num_bits in CODEBOOK, f"Unsupported num_bits={num_bits}, must be one of {list(CODEBOOK.keys())}"
        self.num_bits = num_bits
        self.dtype = dtype
        self.entries = CODEBOOK[num_bits]
        self.elems_per_int32 = 32 // num_bits  # e.g. 8 for 4-bit
        self.mask = (1 << num_bits) - 1

    @cute.jit
    def unpack(self, packed_val: Int32, index: Int32) -> Int32:
        shift = index * self.num_bits
        return (packed_val >> shift) & self.mask

    @cute.jit
    def codebook_lookup(self, code: Int32):
        """Compile-time unrolled codebook lookup via if/else chain."""
        result = self.dtype(self.entries[0])
        for c in cutlass.range_constexpr(len(self.entries)):
            if code == c:
                result = self.dtype(self.entries[c])
        return result

    @cute.jit
    def dequantize(self, packed_data: cute.Tensor, output: cute.Tensor, num_int32s: Int32):
        """Dequantize packed Int32 data via codebook lookup.

        Args:
            packed_data: Register tensor of packed Int32 values.
            output: Register tensor to write dequantized fp16/bf16 values.
            num_int32s: Number of Int32 elements in packed_data.
        """
        # Fully unrolled: iterate over each Int32, unpack all sub-elements
        for i in cutlass.range_constexpr(num_int32s):
            packed_val = packed_data[i]
            for j in cutlass.range_constexpr(self.elems_per_int32):
                code = self.unpack(packed_val, j)
                out_idx = i * self.elems_per_int32 + j
                output[out_idx] = self.codebook_lookup(code)
