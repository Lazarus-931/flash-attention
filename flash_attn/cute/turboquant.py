
import cutlass  # type: ignore
import cutlass.cute as cute  # type: ignore
from cutlass import Int32, Float32  # type: ignore

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
        assert num_bits in CODEBOOK
        self.num_bits = num_bits
        self.dtype = dtype
        self.entries = CODEBOOK[num_bits]
        self.elems_per_int32 = 32 // num_bits
        self.mask = (1 << num_bits) - 1
        self.num_entries = len(self.entries)

    def make_codebook(self):
        cb = cute.make_rmem_tensor((self.num_entries,), self.dtype)
        for i in cutlass.range_constexpr(self.num_entries):
            cb[i] = self.dtype(self.entries[i])
        return cb

    @cute.jit
    def dequantize(self, packed_data: cute.Tensor, output: cute.Tensor,
                   num_int32s: Int32, cb: cute.Tensor):
        for i in cutlass.range_constexpr(num_int32s):
            packed_val = packed_data[i]
            for j in cutlass.range_constexpr(self.elems_per_int32):
                code = (packed_val >> (j * self.num_bits)) & self.mask
                output[i * self.elems_per_int32 + j] = cb[code]
