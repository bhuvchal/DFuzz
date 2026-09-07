#pragma once
#include "dfuzz/case.h"
#include <cstdint>
#include <vector>

namespace dfuzz {

// --- Repacked (interleaved) weight kernels --------------------------------
// gemv/gemm take weights pre-interleaved across several rows so the SIMD path
// can produce multiple output columns at once. That interleaving is pure index
// arithmetic and is a materially larger surface than the single-row dot
// products: mxfp4 4x4, q4_K 8x8, each with a _generic twin.
//
// Three comparison axes, as before, plus one that is specific to this surface
// and stronger than any of them:
//
//   gemv row j  ==  plain vec_dot of row j
//
// The repacked kernel and the scalar dot product must agree row for row. That
// makes the interleaving itself falsifiable without any reference of ours.

// Interleaves nrows mxfp4-quantized rows into block_mxfp4x4 form.
// Mirrors make_block_mxfp4x4() with blck_size_interleave == 4.
void repack_mxfp4_4x4(const void* src_rows, int nrows, int nblocks,
                      std::vector<uint8_t>& out);

struct GemvResult {
    std::vector<float> simd;      // one value per output column
    std::vector<float> generic;
    std::vector<float> vecdot;    // per-row plain dot product
    std::vector<double> reference;// fp64
    std::vector<double> intermediate_mag;
};

// Runs all four for an nc-column, n-deep mxfp4 gemv.
GemvResult run_mxfp4_gemv(const std::vector<std::vector<float>>& rows,
                          const std::vector<float>& activations);

// Interleaves 8 q4_K rows into block_q4_Kx8. Substantially more intricate than
// the mxfp4 case: 8-byte round-robin for the quants, plus a re-pack of the
// 6-bit scale/min pairs across all 8 rows with the high and low bits split
// between separate output bytes. Mirrors make_block_q4_Kx8(.., 8).
void repack_q4_K_8x8(const void* src_rows, int nrows, int nblocks,
                     std::vector<uint8_t>& out);

// nc must be a multiple of 8; n a multiple of 256.
GemvResult run_q4_K_gemv(const std::vector<std::vector<float>>& rows,
                         const std::vector<float>& activations);

} // namespace dfuzz
