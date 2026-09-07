#include "dfuzz/generate.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace dfuzz {

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
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::normal_distribution<float>       g(0.0f, 1.0f);

    switch (spec.cls) {
        case InputClass::Uniform:
            for (int i = 0; i < n; ++i) { c.x[i] = u(rng); c.y[i] = u(rng); }
            break;

        case InputClass::Cancellation: {
            // Build pairs whose products cancel: x[2k]*y[2k] = -x[2k+1]*y[2k+1].
            // The true dot product collapses toward zero while sum|x_i y_i|
            // does not, driving the condition number up. This is where
            // relative-to-result error explodes with no bug present, and the
            // reason the bound is expressed against abs_sum.
            for (int i = 0; i + 1 < n; i += 2) {
                const float a = g(rng), b = g(rng);
                c.x[i] = a;      c.y[i] = b;
                c.x[i+1] = a;    c.y[i+1] = -b;
            }
            break;
        }

        case InputClass::MixedMagnitude: {
            std::uniform_int_distribution<int> e(-30, 30);
            for (int i = 0; i < n; ++i) {
                c.x[i] = std::ldexp(g(rng), e(rng));
                c.y[i] = std::ldexp(g(rng), e(rng));
            }
            break;
        }

        case InputClass::Denormal: {
            std::uniform_int_distribution<int> s(0, 1);
            const float tiny = std::numeric_limits<float>::denorm_min();
            std::uniform_int_distribution<int> mult(1, 1000);
            for (int i = 0; i < n; ++i) {
                c.x[i] = (s(rng) ? 1.f : -1.f) * tiny * (float)mult(rng);
                c.y[i] = (s(rng) ? 1.f : -1.f) * tiny * (float)mult(rng);
            }
            break;
        }

        case InputClass::SignedZero: {
            std::uniform_int_distribution<int> s(0, 2);
            for (int i = 0; i < n; ++i) {
                const int a = s(rng), b = s(rng);
                c.x[i] = a == 0 ? 0.0f : (a == 1 ? -0.0f : g(rng));
                c.y[i] = b == 0 ? 0.0f : (b == 1 ? -0.0f : g(rng));
            }
            break;
        }

        case InputClass::PowerOfTwo: {
            std::uniform_int_distribution<int> e(-10, 10), s(0, 1);
            for (int i = 0; i < n; ++i) {
                c.x[i] = std::ldexp(s(rng) ? 1.f : -1.f, e(rng));
                c.y[i] = std::ldexp(s(rng) ? 1.f : -1.f, e(rng));
            }
            break;
        }

        case InputClass::Saturating: {
            // One huge value per block hijacks the shared scale, crushing the
            // rest toward zero. Exercises the encoder's saturation path and
            // the resulting all-but-one-zero block in the kernel.
            const int be = block_elems(spec.fmt);
            for (int i = 0; i < n; ++i) { c.x[i] = g(rng) * 1e-3f; c.y[i] = g(rng); }
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
            std::uniform_int_distribution<int> pos(0, n - 1);
            const int p = pos(rng);
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
