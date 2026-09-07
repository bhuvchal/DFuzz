#pragma once
#include "dfuzz/case.h"
#include <cstdint>
#include <random>
#include <string_view>

namespace dfuzz {

// Input classes, biased toward the seams. Uniform random inputs find close to
// nothing: the failure modes hide in cancellation, denormals, mixed magnitudes,
// exact ties, and shapes one element off a block boundary.
enum class InputClass {
    Uniform,        // control: the distribution that finds nothing
    Cancellation,   // pairs whose products nearly cancel -> cond explodes
    MixedMagnitude, // values spanning many orders within one block
    Denormal,       // subnormals, flush-to-zero divergence
    SignedZero,     // +0 / -0 mixtures
    PowerOfTwo,     // exact ties in rounding
    Saturating,     // one value dominates the block scale, crushing the rest
    AlternatingMax, // +max/-max alternation: worst case for accumulation order
    SingleSpike,    // one large value among zeros
    Count
};

const char* input_class_name(InputClass c);

// A fully reproducible generation request. Everything the fuzzer varies lives
// here, so a finding is replayable from this struct plus the seed alone.
struct GenSpec {
    Format     fmt   = Format::mxfp4;
    InputClass cls   = InputClass::Uniform;
    int        blocks = 1;
    uint64_t   seed  = 0;
    // Offset the operand pointers into an oversized buffer. Exercises the
    // kernels' alignment assumptions; 0 means naturally aligned.
    int        align_offset = 0;
};

// Deterministic: the same spec always yields the same case.
Case generate(const GenSpec& spec);

// Scales any block that exceeds the format's representable magnitude back into
// domain. Applied by generate(); exposed so the shrinker's candidates can be
// kept in domain too.
void clamp_to_domain(Case& c);

} // namespace dfuzz
