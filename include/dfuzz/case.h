#pragma once
#include "dfuzz/formats.h"
#include <cstdint>
#include <vector>

namespace dfuzz {

// A reproducible test case: the fp32 inputs, plus everything needed to replay
// it without the fuzzer that produced it.
struct Case {
    Format             fmt  = Format::mxfp4;
    std::vector<float> x;          // weight-side operand, pre-quantization
    std::vector<float> y;          // activation-side operand
    uint64_t           seed = 0;   // provenance only; replay uses x/y directly

    int n() const { return (int)x.size(); }
    int blocks() const { return n() / block_elems(fmt); }
};

// Which pair of implementations diverged. Kept as a distinct axis because the
// triage meaning differs: SimdVsGeneric involves no reference of ours, so a
// failure there cannot be our own construction error.
enum class Axis { SimdVsRef, GenericVsRef, SimdVsGeneric };
const char* axis_name(Axis a);

// What a divergence looked like. A value type with equality so "is this still
// the SAME bug?" is one comparison, and so adding a field forces every
// comparison site to be revisited by the compiler rather than by memory.
//
// This is the guard against the shrinker walking from one bug into a different
// one and reporting the wrong thing.
struct Signature {
    Format fmt  = Format::mxfp4;
    Axis   axis = Axis::SimdVsRef;
    // Order of magnitude of the bound violation, not the raw ratio: shrinking
    // legitimately moves the ratio around, so requiring exact equality would
    // reject good reductions. Requiring the same decade is the useful middle.
    int    severity_decade = 0;
    // Distinguishes "returned garbage" from "accumulated too much error".
    bool   nonfinite = false;

    friend bool operator==(const Signature& a, const Signature& b) {
        return a.fmt == b.fmt && a.axis == b.axis &&
               a.severity_decade == b.severity_decade &&
               a.nonfinite == b.nonfinite;
    }
    friend bool operator!=(const Signature& a, const Signature& b) { return !(a == b); }
};

Signature make_signature(Format fmt, Axis axis, double ratio, bool nonfinite);

} // namespace dfuzz
