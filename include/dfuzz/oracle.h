#pragma once
#include "dfuzz/case.h"
#include "dfuzz/bound.h"
#include <optional>
#include <vector>

namespace dfuzz {

struct Outcome {
    bool      failed = false;
    Signature sig{};
    double    ratio = 0.0;    // worst ratio across axes
    Axis      worst = Axis::SimdVsRef;
    double    simd = 0, generic = 0, reference = 0, abs_sum = 0;
};

// Evaluates a Case against all three axes.
//
// Holds its own scratch buffers and reuses them across calls.
//
// Measured (tests/bench_oracle.cpp, nvfp4 n=1024, 20k evaluations): reuse is
// only ~1.04x faster than allocating per call -- the kernels and the fp64
// reference dominate at ~15us/eval, and allocation is noise beside them. The
// speed argument does not survive measurement and is not claimed.
//
// What reuse does buy is bounded memory behaviour: 2 buffer growths across
// 20k evaluations instead of 40k allocate/free pairs, so a long fuzzing run
// has a flat allocation profile and no fragmentation growth. That is the
// honest reason this is here.
//
// Non-copyable so a stale copy of the scratch state cannot be handed around;
// movable so it can be returned and stored by value.
class Oracle {
public:
    Oracle();
    ~Oracle();
    Oracle(const Oracle&)            = delete;
    Oracle& operator=(const Oracle&) = delete;
    Oracle(Oracle&&) noexcept;
    Oracle& operator=(Oracle&&) noexcept;

    Outcome evaluate(const Case& c);

    // Cheap predicate used by the shrink loop: does this case still exhibit
    // the same failure? Requiring signature equality is what stops the
    // shrinker from walking into a different bug.
    bool still_fails(const Case& c, const Signature& want);

    long long evaluations() const { return evals_; }
    long long allocations()  const { return allocs_; }

private:
    struct Scratch;
    Scratch*  s_     = nullptr;
    long long evals_ = 0;
    long long allocs_ = 0;
};

} // namespace dfuzz
