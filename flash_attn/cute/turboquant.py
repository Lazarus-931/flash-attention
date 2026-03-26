
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

    @cute.jit
    def unpack(self, packed_val: Int32, index: Int32) -> Int32:
        shift = index * self.num_bits
        mask = (1 << self.num_bits) - 1
        return (packed_val >> shift) & mask

    @cute.jit
    def dequantize(self, packed_data: cute.Tensor, output: cute.Tensor, num_elements: Int32):
        """Dequantize packed low-bit data via codebook lookup.

        Args:
            packed_data: Register tensor of packed uint8 values.
            output: Register tensor to write dequantized fp16/bf16 values.
            num_elements: Number of elements to dequantize.
        """
        elems_per_pack = 8 // self.num_bits


        cb = cute.make_rmem_tensor((len(self.entries),), Float32)
        for i in cutlass.range_constexpr(len(self.entries)):
            cb[i] = self.entries[i]

        for i in cutlass.range(num_elements, unroll=1):
            pack_idx = i // elems_per_pack
            elem_idx = i % elems_per_pack
            code = self.unpack(packed_data[pack_idx], elem_idx)
            output[i] = cb[code]
