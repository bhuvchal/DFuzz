#pragma once
#include <cstdint>
#include <vector>

namespace dfuzz {

// --- Recent, less-trodden formats -----------------------------------------
// mxfp4 / nvfp4 store VALUES as 4-bit floats (E2M1) rather than integers, and
// store SCALES as floats too. ggml keeps the E2M1 table pre-doubled as int8
// {0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}, folding the factor of 2 into the scale
// converters -- so inner products remain exact int32 accumulation despite the
// format being nominally floating-point.

constexpr int QK_MXFP4     = 32;
constexpr int QK_NVFP4     = 64;
constexpr int QK_NVFP4_SUB = 16;
constexpr int QK8_0        = 32;

// E2M1 values, pre-doubled. Mirrors kvalues_fp4 in ggml-common.h.
inline const int8_t* kvalues_fp4() {
    static const int8_t v[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12};
    return v;
}

// E8M0: a bare 8-bit exponent -- the represented value is a pure power of two,
// so multiplying by it is EXACT in fp32 (barring over/underflow). ggml's
// _HALF variant folds in the 0.5 that undoes the doubled E2M1 table.
// Exactness here is a real, exploitable property: it removes one rounding
// from every mxfp4 block and is why mxfp4's bound is tighter than nvfp4's.
double e8m0_to_double_half(uint8_t e);

// UE4M3: 4 exponent bits (bias 7), 3 mantissa bits, unsigned. Has a mantissa,
// so unlike E8M0 the scale multiply is NOT exact.
double ue4m3_to_double(uint8_t x);

// Exact dequantization to fp64.
void dequantize_mxfp4(const void* vx, int nblocks, std::vector<double>& out);
void dequantize_nvfp4(const void* vx, int nblocks, std::vector<double>& out);
void dequantize_q8_0 (const void* vy, int nblocks, std::vector<double>& out);

// --- Per-format accumulation depth ----------------------------------------
// The depth is where the fp32 adds happen, and it differs per format. Reusing
// one depth across formats is wrong for at least two of them:
//
//   q4_K   : one fp32 add per 256-element super-block  -> n/256
//   mxfp4  : one fp32 add per 32-element block         -> n/32
//   nvfp4  : one fp32 add per 16-element SUB-block     -> n/16
//
// so at equal n, nvfp4 tolerates 16x the accumulation error of q4_K.
enum class Format { q4_K, q6_K, mxfp4, nvfp4 };

int  accumulation_depth(Format f, int n);
const char* format_name(Format f);
// Bytes per block, and elements per block, for buffer sizing.
int  block_bytes(Format f);
int  block_elems(Format f);

// Largest per-block magnitude a format can represent before its stored scale
// overflows and the block round-trips to NaN/Inf.
//
// q4_K and q6_K keep the super-block scale in fp16 (max 65504), so a block
// whose values exceed roughly that times the quant range cannot be encoded:
// 1/iscale overflows to inf, and dequantization yields inf*0 = NaN. Measured
// by bisection, not assumed: q4_K ~1.2e6, q6_K ~7.5e7.
//
// mxfp4 and nvfp4 have no such limit -- E8M0 and UE4M3 are exponent-based and
// track magnitude freely. This is the same property that makes mxfp4's scale
// multiply exact, showing up as dynamic range.
//
// Inputs beyond the limit are OUT OF DOMAIN for the format, not bugs. A
// generator that produces them manufactures false positives.
double max_block_magnitude(Format f);

} // namespace dfuzz
