// Step two, part one: validate dequantization ALONE, before any dot product.
//
// Two bugs at once is what makes this phase take a week instead of a day. So
// this binary compares our fp64 unpacking against ggml's own dequantize_row_*
// element by element, and nothing else.
//
// The predicate is exactness, not closeness. ggml dequantizes to fp32 by
// multiplying fp32 quantities; we do the same arithmetic in fp64. Every ggml
// intermediate is exactly representable in fp64, so rounding our result to
// fp32 must reproduce theirs BIT FOR BIT. Any mismatch is a layout or formula
// error on our side -- never a tolerance question.
#include "dfuzz/kernels.h"
#include "dfuzz/reference.h"
#include "dfuzz/buildinfo.h"
#include "dfuzz/init.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

extern "C" {
void dequantize_row_q4_K(const void* x, float* y, int64_t k);
void dequantize_row_q6_K(const void* x, float* y, int64_t k);
}

using namespace dfuzz;

namespace {

int check_one(const char* label,
              const std::vector<float>& src,
              int nblocks,
              void (*quantize)(const float*, void*, int64_t),
              void (*ggml_dequant)(const void*, float*, int64_t),
              void (*our_dequant)(const void*, int, std::vector<double>&),
              size_t block_bytes) {
    const int n = nblocks * QK_K;
    std::vector<uint8_t> q((size_t)nblocks * block_bytes);
    quantize(src.data(), q.data(), n);

    std::vector<float> theirs(n);
    ggml_dequant(q.data(), theirs.data(), n);

    std::vector<double> ours;
    our_dequant(q.data(), nblocks, ours);

    if ((int)ours.size() != n) {
        std::printf("  %-6s FAIL: size %zu != %d\n", label, ours.size(), n);
        return 1;
    }

    int mismatches = 0;
    double worst = 0.0;
    int worst_idx = -1;
    for (int i = 0; i < n; ++i) {
        const float ours_f32 = (float)ours[i];
        // Bitwise comparison: catches signed-zero and NaN differences that
        // == would silently accept.
        uint32_t a, b;
        std::memcpy(&a, &ours_f32,  4);
        std::memcpy(&b, &theirs[i], 4);
        if (a != b) {
            const double d = std::fabs((double)ours_f32 - (double)theirs[i]);
            if (d > worst || worst_idx < 0) { worst = d; worst_idx = i; }
            if (++mismatches <= 4) {
                std::printf("  %-6s  [%4d] ours=%.9g (0x%08x)  theirs=%.9g (0x%08x)\n",
                            label, i, (double)ours_f32, a, (double)theirs[i], b);
            }
        }
    }

    if (mismatches) {
        std::printf("  %-6s FAIL: %d/%d elements differ, worst |d|=%.4g at [%d]\n",
                    label, mismatches, n, worst, worst_idx);
        return 1;
    }
    std::printf("  %-6s ok: %d/%d elements bit-identical to ggml\n", label, n, n);
    return 0;
}

// Input classes chosen to exercise the encoders' corners, not to be "typical".
std::vector<float> make_input(const char* kind, int n, std::mt19937& rng) {
    std::vector<float> v(n);
    if (!std::strcmp(kind, "normal")) {
        std::normal_distribution<float> d(0.f, 1.f);
        for (auto& e : v) e = d(rng);
    } else if (!std::strcmp(kind, "allzero")) {
        std::fill(v.begin(), v.end(), 0.f);
    } else if (!std::strcmp(kind, "mixedmag")) {
        std::uniform_int_distribution<int> e(-30, 30);
        std::normal_distribution<float> m(0.f, 1.f);
        for (auto& x : v) x = std::ldexp(m(rng), e(rng));
    } else if (!std::strcmp(kind, "pow2")) {
        std::uniform_int_distribution<int> e(-8, 8);
        std::uniform_int_distribution<int> s(0, 1);
        for (auto& x : v) x = std::ldexp(s(rng) ? 1.f : -1.f, e(rng));
    } else if (!std::strcmp(kind, "onebig")) {
        // One huge value dominates the block scale, crushing everything else
        // to zero -- exercises the saturation path in the encoder.
        std::normal_distribution<float> d(0.f, 1.f);
        for (auto& e : v) e = d(rng);
        v[0] = 1e30f;
    } else if (!std::strcmp(kind, "denormal")) {
        std::uniform_int_distribution<int> s(0, 1);
        for (auto& x : v) x = (s(rng) ? 1.f : -1.f) * 1e-42f;
    }
    return v;
}

} // namespace

int main() {
    dfuzz::Guard guard;   // ggml lookup tables must be live before any kernel call
    print_build_config();
    std::printf("\n# dequantization validation (must be bit-exact vs ggml)\n");

    const char* kinds[] = {"normal", "allzero", "mixedmag", "pow2", "onebig", "denormal"};
    const int nblocks = 3;
    const int n = nblocks * QK_K;

    int failures = 0;
    std::mt19937 rng(0xC0FFEE);
    for (const char* k : kinds) {
        std::printf("input=%s\n", k);
        auto src = make_input(k, n, rng);
        failures += check_one("q4_K", src, nblocks, quantize_row_q4_K_ref,
                              dequantize_row_q4_K, dequantize_q4_K, 144);
        failures += check_one("q6_K", src, nblocks, quantize_row_q6_K_ref,
                              dequantize_row_q6_K, dequantize_q6_K, 210);
    }

    // q8_K has no ggml dequantizer (it is an intermediate format), so validate
    // against its definition directly: y = d * qs, and confirm the bsums we
    // deliberately ignore are consistent with qs.
    std::printf("input=normal\n");
    {
        auto src = make_input("normal", n, rng);
        std::vector<uint8_t> q((size_t)nblocks * 292);
        quantize_row_q8_K_ref(src.data(), q.data(), n);
        std::vector<double> ours;
        dequantize_q8_K(q.data(), nblocks, ours);
        int bad = 0, bad_bsum = 0;
        for (int b = 0; b < nblocks; ++b) {
            const uint8_t* blk = q.data() + (size_t)b * 292;
            float d; std::memcpy(&d, blk, 4);
            const int8_t* qs = (const int8_t*)(blk + 4);
            const int16_t* bs = (const int16_t*)(blk + 4 + QK_K);
            for (int i = 0; i < QK_K; ++i)
                if ((float)ours[(size_t)b*QK_K + i] != d * (float)qs[i]) ++bad;
            for (int j = 0; j < QK_K/16; ++j) {
                int sum = 0;
                for (int ii = 0; ii < 16; ++ii) sum += qs[j*16 + ii];
                if (sum != bs[j]) ++bad_bsum;
            }
        }
        std::printf("  %-6s %s: %d elements, bsums consistent=%s\n", "q8_K",
                    bad ? "FAIL" : "ok", n, bad_bsum ? "NO" : "yes");
        if (bad) ++failures;
    }

    std::printf(failures ? "\nDEQUANT VALIDATION FAILED (%d)\n"
                         : "\nDEQUANT VALIDATION PASSED\n", failures);
    return failures ? 1 : 0;
}
