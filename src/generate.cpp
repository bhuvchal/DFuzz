#include "dfuzz/generate.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace dfuzz {
namespace {

// std::mt19937_64 is specified exactly by the standard, but the DISTRIBUTIONS
// are not: normal_distribution and uniform_real_distribution may use different
// algorithms in libstdc++ and libc++, so the same seed yields different values
// on different platforms.
//
// That silently broke the cross-path comparison -- two builds were never
// running the same inputs, and a divergence between them meant nothing. It
// also produced phantom findings on x86 that could not be reproduced on arm64
// from the same seed.
//
// These generate values from the engine directly, so a seed denotes the same
// input everywhere.

// Uniform in [-1, 1), from the top 24 bits (fp32 has 24 significand bits).
float portable_uniform(std::mt19937_64& rng) {
    const uint32_t bits = (uint32_t)(rng() >> 40);          // 24 bits
    return (float)((double)bits / 8388608.0 - 1.0);         // /2^23 - 1
}

// Standard normal via Box-Muller. Deterministic given the engine, and uses a
// fixed number of engine draws so the stream stays aligned across platforms.
float portable_normal(std::mt19937_64& rng) {
    // Avoid log(0): map to (0,1].
    const double u1 = ((double)(rng() >> 11) + 1.0) * (1.0 / 9007199254740993.0);
    const double u2 = ((double)(rng() >> 11)      ) * (1.0 / 9007199254740992.0);
    return (float)(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
}

// Uniform integer in [lo, hi], inclusive, without std::uniform_int_distribution.
int portable_int(std::mt19937_64& rng, int lo, int hi) {
    const uint64_t span = (uint64_t)(hi - lo) + 1u;
    return lo + (int)(rng() % span);
}

} // namespace

const char* input_class_name(InputClass c) {
    switch (c) {
        case InputClass::Uniform:        return "uniform";
        case InputClass::Cancellation:   return "cancellation";
        case InputClass::MixedMagnitude: return "mixed-magnitude";
        case InputClass::Denormal:       return "denormal";
        case InputClass::SignedZero:     return "signed-zero";
        case InputClass::PowerOfTwo:     return "power-of-two";
        case InputClass::Saturating:     return "saturating";
        case InputClass::AlternatingMax: return "alternating-max";
        case InputClass::SingleSpike:    return "single-spike";
        default:                         return "?";
    }
}

Case generate(const GenSpec& spec) {
    Case c;
    c.fmt  = spec.fmt;
    c.seed = spec.seed;

    const int n = spec.blocks * block_elems(spec.fmt);
    c.x.assign(n, 0.0f);
    c.y.assign(n, 0.0f);

    std::mt19937_64 rng(spec.seed);
    auto u = [&rng]{ return portable_uniform(rng); };
    auto g = [&rng]{ return portable_normal(rng); };

    switch (spec.cls) {
        case InputClass::Uniform:
            for (int i = 0; i < n; ++i) {
                const float a = u();
                const float b = u();
                c.x[i] = a; c.y[i] = b;
            }
            break;

        case InputClass::Cancellation: {
            // Build pairs whose products cancel: x[2k]*y[2k] = -x[2k+1]*y[2k+1].
            // The true dot product collapses toward zero while sum|x_i y_i|
            // does not, driving the condition number up. This is where
            // relative-to-result error explodes with no bug present, and the
            // reason the bound is expressed against abs_sum.
            for (int i = 0; i + 1 < n; i += 2) {
                const float a = g(), b = g();
                c.x[i] = a;      c.y[i] = b;
                c.x[i+1] = a;    c.y[i+1] = -b;
            }
            break;
        }

        case InputClass::MixedMagnitude: {
            // Each draw is sequenced explicitly. Function-argument evaluation
            // order is UNSPECIFIED in C++, so writing
            //   ldexp(g(), portable_int(rng, ...))
            // lets two platforms consume the engine stream in opposite orders
            // and produce different inputs from the same seed -- which silently
            // invalidates the whole cross-path comparison.
            for (int i = 0; i < n; ++i) {
                const float mx = g();
                const int   ex = portable_int(rng, -30, 30);
                c.x[i] = std::ldexp(mx, ex);
                const float my = g();
                const int   ey = portable_int(rng, -30, 30);
                c.y[i] = std::ldexp(my, ey);
            }
            break;
        }

        case InputClass::Denormal: {
            const float tiny = std::numeric_limits<float>::denorm_min();
            for (int i = 0; i < n; ++i) {
                const int sx = portable_int(rng,0,1);
                const int mx = portable_int(rng,1,1000);
                c.x[i] = (sx ? 1.f : -1.f) * tiny * (float)mx;
                const int sy = portable_int(rng,0,1);
                const int my = portable_int(rng,1,1000);
                c.y[i] = (sy ? 1.f : -1.f) * tiny * (float)my;
            }
            break;
        }

        case InputClass::SignedZero: {
            for (int i = 0; i < n; ++i) {
                const int a = portable_int(rng,0,2);
                const int b = portable_int(rng,0,2);
                c.x[i] = a == 0 ? 0.0f : (a == 1 ? -0.0f : g());
                c.y[i] = b == 0 ? 0.0f : (b == 1 ? -0.0f : g());
            }
            break;
        }

        case InputClass::PowerOfTwo: {
            for (int i = 0; i < n; ++i) {
                const int sx = portable_int(rng,0,1);
                const int ex = portable_int(rng,-10,10);
                c.x[i] = std::ldexp(sx ? 1.f : -1.f, ex);
                const int sy = portable_int(rng,0,1);
                const int ey = portable_int(rng,-10,10);
                c.y[i] = std::ldexp(sy ? 1.f : -1.f, ey);
            }
            break;
        }

        case InputClass::Saturating: {
            // One huge value per block hijacks the shared scale, crushing the
            // rest toward zero. Exercises the encoder's saturation path and
            // the resulting all-but-one-zero block in the kernel.
            const int be = block_elems(spec.fmt);
            for (int i = 0; i < n; ++i) {
                const float a = g();
                const float b = g();
                c.x[i] = a * 1e-3f; c.y[i] = b;
            }
            for (int b = 0; b < spec.blocks; ++b) c.x[b * be] = 1e4f;
            break;
        }

        case InputClass::AlternatingMax: {
            // +M, -M, +M, ... : maximal opportunity for summation order to
            // matter, since every partial sum swings the full magnitude.
            const float M = 6.0f; // the largest E2M1 magnitude
            for (int i = 0; i < n; ++i) {
                c.x[i] = (i % 2) ? -M : M;
                c.y[i] = (i % 4 < 2) ? 1.0f : -1.0f;
            }
            break;
        }

        case InputClass::SingleSpike: {
            const int p = portable_int(rng, 0, n - 1);
            c.x[p] = 1.0f; c.y[p] = 1.0f;
            break;
        }

        default: break;
    }

    // Clamp into the format's representable domain. Values beyond it make the
    // encoder's fp16 scale overflow, so the block round-trips to NaN and every
    // comparison "fails" -- a false positive manufactured by the generator,
    // not a kernel defect. Clamping per block keeps the intended distribution
    // shape while staying in domain.
    clamp_to_domain(c);
    return c;
}

namespace {

// Rescale any block of `v` whose peak magnitude exceeds `cap`.
void clamp_blocks(std::vector<float>& v, int be, double cap) {
    if (be <= 0 || !(cap < HUGE_VAL)) return;
    for (int b = 0; b * be < (int)v.size(); ++b) {
        const int hi = std::min((int)v.size(), (b + 1) * be);
        float mx = 0.0f;
        for (int i = b * be; i < hi; ++i) {
            const float a = std::fabs(v[(size_t)i]);
            if (a > mx) mx = a;
        }
        if (mx > cap) {
            const float k = (float)(cap / (double)mx);
            for (int i = b * be; i < hi; ++i) v[(size_t)i] *= k;
        }
    }
}

} // namespace

void clamp_to_domain(Case& c) {
    // The WEIGHT side is limited by the format's own scale representation:
    // fp16 for the K-quants, unlimited for the exponent-scaled fp4 formats.
    clamp_blocks(c.x, block_elems(c.fmt), max_block_magnitude(c.fmt) * 0.25);

    // The ACTIVATION side is q8_0 (32-element blocks) or q8_K (256), and BOTH
    // store their scale in fp16 -- so the activation operand has a domain limit
    // even when the weight format does not. Missing this left mxfp4 and nvfp4
    // producing NaN from perfectly ordinary weights.
    const bool q8K = (c.fmt == Format::q4_K || c.fmt == Format::q6_K);
    clamp_blocks(c.y, q8K ? 256 : 32, 65504.0 * 0.25);
}

} // namespace dfuzz
