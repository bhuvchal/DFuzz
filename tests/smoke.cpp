// Week 1-2 milestone: prove the chain links and runs end to end.
//   floats -> ggml quantizers -> {NEON kernel, generic kernel, fp64 reference}
// and that all three agree within the derived bound on benign input.
#include "dfuzz/kernels.h"
#include "dfuzz/reference.h"
#include "dfuzz/bound.h"
#include "dfuzz/buildinfo.h"
#include "dfuzz/init.h"
#include <cstdio>
#include <random>
#include <vector>

using namespace dfuzz;

int main() {
    dfuzz::Guard guard;   // ggml lookup tables must be live before any kernel call
    print_build_config();
    std::printf("\n");
    const int nblocks = 4;
    const int n = nblocks * QK_K;

    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> xf(n), yf(n);
    for (int i = 0; i < n; ++i) { xf[i] = dist(rng); yf[i] = dist(rng); }

    // ggml block sizes, in bytes.
    std::vector<uint8_t> xq((size_t)nblocks * 144);  // sizeof(block_q4_K)
    std::vector<uint8_t> yq((size_t)nblocks * 292);  // sizeof(block_q8_K)
    quantize_row_q4_K_ref(xf.data(), xq.data(), n);
    quantize_row_q8_K_ref(yf.data(), yq.data(), n);

    float s_simd = 0.0f, s_gen = 0.0f;
    ggml_vec_dot_q4_K_q8_K        (n, &s_simd, 0, xq.data(), 0, yq.data(), 0, 1);
    ggml_vec_dot_q4_K_q8_K_generic(n, &s_gen,  0, xq.data(), 0, yq.data(), 0, 1);

    std::vector<double> xd, yd;
    dequantize_q4_K(xq.data(), nblocks, xd);
    dequantize_q8_K(yq.data(), nblocks, yd);
    const RefResult ref = reference_dot(xd, yd);

    std::printf("n=%d  nblocks=%d\n", n, nblocks);
    std::printf("  simd      = %.9g\n", (double)s_simd);
    std::printf("  generic   = %.9g\n", (double)s_gen);
    std::printf("  reference = %.9g\n", ref.dot);
    std::printf("  abs_sum   = %.9g   cond = %.4g\n",
                ref.abs_sum, ref.abs_sum / (ref.dot < 0 ? -ref.dot : ref.dot));

    int failures = 0;
    struct { const char* name; double val; } cases[] = {
        {"simd    vs reference", (double)s_simd},
        {"generic vs reference", (double)s_gen},
    };
    for (auto& c : cases) {
        const Verdict v = check(c.val, ref.dot, ref.abs_sum, n);
        std::printf("  %s: obs=%.4g allowed=%.4g ratio=%.4f %s\n",
                    c.name, v.observed, v.allowed, v.ratio, v.ok ? "OK" : "FAIL");
        if (!v.ok) ++failures;
    }
    // simd vs generic: pure conformance, no reference involved.
    const Verdict vg = check((double)s_simd, (double)s_gen, ref.abs_sum, n);
    std::printf("  simd    vs generic  : obs=%.4g allowed=%.4g ratio=%.4f %s\n",
                vg.observed, vg.allowed, vg.ratio, vg.ok ? "OK" : "FAIL");
    if (!vg.ok) ++failures;

    std::printf(failures ? "\nSMOKE FAILED (%d)\n" : "\nSMOKE PASSED\n", failures);
    return failures ? 1 : 0;
}
