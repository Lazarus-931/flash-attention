
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

ODD_POLY_COEFFS = {
    1: [1.59576912],
    2: [0.05069251, 0.89288693],
    3: [4.07666333e-04, 5.24606083e-03, 4.89310189e-01],
    4: [4.11131823e-07, -9.31039455e-06, 1.14310329e-03, 2.56189071e-01],
}

ODD_POLY_CENTER = {
    1: 0.5,
    2: 1.5,
    3: 3.5,
    4: 7.5,
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
        self.poly_coeffs = ODD_POLY_COEFFS[num_bits]
        self.center = ODD_POLY_CENTER[num_bits]

    @cute.jit
    def dequantize(self, packed_data: cute.Tensor, output: cute.Tensor, num_int32s: Int32):
        center = self.dtype(self.center)
        coeffs = [self.dtype(c) for c in self.poly_coeffs]
        num_coeffs = len(self.poly_coeffs)

        for i in cutlass.range_constexpr(num_int32s):
            packed_val = packed_data[i]
            for j in cutlass.range_constexpr(self.elems_per_int32):
                code = (packed_val >> (j * self.num_bits)) & self.mask
                t = self.dtype(code) - center
                t2 = t * t
                val = coeffs[0]
                for k in cutlass.range_constexpr(1, num_coeffs):
                    val = val * t2 + coeffs[k]
                output[i * self.elems_per_int32 + j] = val * t
