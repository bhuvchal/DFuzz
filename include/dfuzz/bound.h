#pragma once
#include <cstddef>

namespace dfuzz {

// Derived error bounds. See README for the full derivation; the short version:
//
//   |fl(x.y) - x.y| <= gamma_k * sum|x_i y_i|,   gamma_k = k*u / (1 - k*u)
//
// where k is the *effective accumulation depth*, not n. The depth depends on
// how the kernel accumulates, which is the entire reason a single hardcoded
// tolerance is wrong.

// fp32 unit roundoff (2^-24). Kernel-side accumulation is fp32.
constexpr double kUnitRoundoffF32 = 5.9604644775390625e-08;

// FMA contraction.
//
// When the compiler is allowed to fuse a*b+c into a single fused multiply-add,
// the intermediate product is NOT rounded. That is more accurate per operation
// but it changes the result, and two kernels compiled from different source
// shapes contract differently -- so a contracted kernel and an uncontracted one
// disagree by far more than either disagrees with the exact answer.
//
// Measured (CI, GCC 13.3 on x86_64, q4_K 8x8 gemm vs gemv, identical inputs):
//
//   -mavx2 -mfma                     worst ratio 5.07   FAILS
//   -mavx2 -mfma -ffp-contract=off   worst ratio 0.264  clean
//   runtime dispatch (no -march)     worst ratio 0.264  clean
//   arm64 NEON                       worst ratio 0.238  clean
//
// Same instruction set, same CPU, same inputs; only the fusion differs. So
// contraction is a real term in the error model, not an artifact to tune away:
// a bound derived on one contraction setting does not transfer to another.
//
// Each contracted operation removes one rounding but shifts the result by up
// to one ulp of the unrounded product, so the divergence between a contracted
// and an uncontracted implementation is bounded by roughly one extra unit
// roundoff per accumulation step. kContractionSlack scales the depth term to
// cover it.
constexpr double kContractionSlack = 8.0;

// gamma_k, guarding against the k*u >= 1 blowup.
inline double gamma_k(double k, double u = kUnitRoundoffF32) {
    const double ku = k * u;
    return ku >= 1.0 ? 1.0e300 : ku / (1.0 - ku);
}

// Effective depth for block-quantized dot products.
//
// Within a QK_K=256 super-block, int8 x int4 products accumulate *exactly* in
// int32 -- no rounding whatsoever. Rounding enters only when each block's
// integer sum is scaled by (d * d8) and those per-block contributions are
// summed in fp32. So the depth is the number of blocks, not the number of
// elements: a 4096-element dot product has depth 16, not 4096.
//
// The fp16 super-block scale is exact as stored (it is the value the format
// denotes), so it contributes no error against a correctly-built reference.
inline double effective_depth_blocked(int n, int qk = 256) {
    const int nblocks = n / qk;
    return nblocks > 0 ? (double)nblocks : 1.0;
}

// The comparison predicate.
//
// Note the bound scales with abs_sum, NOT with |dot|. Under cancellation the
// true result collapses while abs_sum does not, so relative-to-result error
// explodes with no bug present. Comparing against abs_sum is what makes
// cancellation inputs interesting rather than a false-positive factory.
struct Verdict {
    bool   ok;
    double observed;   // |fast - reference|
    double allowed;    // gamma_k * abs_sum, plus fp32 representation slack
    double ratio;      // observed / allowed; > 1 fails
};

// When both sides may have been compiled with FMA contraction enabled (the
// default for GCC and for clang with -ffp-contract=fast), pass allow_contract
// so the bound covers the extra divergence. Measured to be necessary on x86
// and harmless where contraction does not occur.
Verdict check(double fast, double reference, double abs_sum, int n, int qk = 256,
              bool allow_contract = true);

} // namespace dfuzz
