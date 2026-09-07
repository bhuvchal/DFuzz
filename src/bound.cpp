#include "dfuzz/bound.h"
#include <cmath>

namespace dfuzz {

Verdict check(double fast, double reference, double abs_sum, int n, int qk,
              bool allow_contract) {
    const double depth = effective_depth_blocked(n, qk);

    // Accumulation error, proportional to sum|x_i y_i|. Contraction widens
    // this: a fused and an unfused implementation of the same arithmetic
    // disagree by more than either disagrees with the exact result.
    const double eff_depth = allow_contract ? depth * kContractionSlack : depth;
    double allowed = gamma_k(eff_depth) * abs_sum;

    // The kernel returns fp32, so the result carries one final rounding of the
    // true value regardless of how it was accumulated. Without this term, cases
    // whose abs_sum is tiny would have a near-zero budget and fail spuriously.
    allowed += kUnitRoundoffF32 * std::fabs(reference);

    // Guard the degenerate all-zero case, where both terms vanish and any
    // nonzero difference -- including a signed-zero mismatch -- would fail.
    if (allowed == 0.0) allowed = 0.0;

    const double observed = std::fabs(fast - reference);

    // NaN/Inf never satisfy the comparison; report them as failures explicitly
    // rather than letting a NaN comparison silently return "ok".
    if (std::isnan(fast) || std::isnan(reference) ||
        std::isinf(fast) || std::isinf(reference)) {
        return Verdict{false, observed, allowed, INFINITY};
    }

    const double ratio = allowed > 0.0 ? observed / allowed
                                       : (observed == 0.0 ? 0.0 : INFINITY);
    return Verdict{observed <= allowed, observed, allowed, ratio};
}

} // namespace dfuzz
