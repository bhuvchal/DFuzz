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

Verdict check(double fast, double reference, double abs_sum, int n, int qk = 256);

} // namespace dfuzz
