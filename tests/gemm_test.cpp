// gemm: both operands interleaved, output written through a stride.
//
// Decisive axis is reference-free: gemm output[m][j] must equal gemv output[j]
// computed on activation row m, since both consume the same repacked weights.
#include "dfuzz/gemv.h"
#include "dfuzz/generate.h"
#include "dfuzz/bound.h"
#include "dfuzz/init.h"
#include "dfuzz/buildinfo.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace dfuzz;

int main() {
    dfuzz::Guard guard;
    print_build_config();
    std::printf("\n# gemm: mxfp4 4x4 and q4_K 8x8, both operands interleaved\n");

    const InputClass classes[] = {
        InputClass::Uniform, InputClass::MixedMagnitude, InputClass::PowerOfTwo,
        InputClass::Cancellation, InputClass::Denormal, InputClass::SignedZero,
        InputClass::Saturating, InputClass::AlternatingMax, InputClass::SingleSpike,
    };
    const int nrs[] = {4, 8, 12, 16};
    const int ncs_mx[] = {4, 8, 16, 32};
    const int ncs_q4[] = {8, 16, 24, 32};
    const int nbs[] = {1, 2, 3, 5, 8};

    long long cases = 0;
    int failures = 0;
    double w_gg = 0, w_sg = 0, w_rf = 0;

  for (int fmt_i = 0; fmt_i < 2; ++fmt_i) {
    const bool is_q4 = (fmt_i == 1);
    const Format fmt = is_q4 ? Format::q4_K : Format::mxfp4;
    const int be = block_elems(fmt);
    const int* ncs = is_q4 ? ncs_q4 : ncs_mx;
    double f_gg = 0, f_sg = 0, f_rf = 0; long long fcases = 0;

    for (InputClass cls : classes)
    for (int nr : nrs)
    for (int ci = 0; ci < 4; ++ci)
    for (int nblocks : nbs)
    for (int seed = 0; seed < 4; ++seed) {
        const int nc = ncs[ci];
        const int n = nblocks * be;
        std::vector<std::vector<float>> wrows((size_t)nc), arows((size_t)nr);
        for (int j = 0; j < nc; ++j) {
            GenSpec sp; sp.fmt = fmt; sp.cls = cls; sp.blocks = nblocks;
            sp.seed = 0xC0FFEEull ^ ((uint64_t)cls<<40) ^ ((uint64_t)nc<<28)
                    ^ ((uint64_t)nblocks<<16) ^ ((uint64_t)seed<<8) ^ (uint64_t)j;
            wrows[(size_t)j] = generate(sp).x;
        }
        for (int m = 0; m < nr; ++m) {
            GenSpec sp; sp.fmt = fmt; sp.cls = cls; sp.blocks = nblocks;
            sp.seed = 0xBEEF00ull ^ ((uint64_t)cls<<36) ^ ((uint64_t)nr<<24)
                    ^ ((uint64_t)nblocks<<12) ^ ((uint64_t)seed<<4) ^ (uint64_t)m;
            arows[(size_t)m] = generate(sp).y;
        }

        const GemmResult g = is_q4 ? run_q4_K_gemm(wrows, arows)
                                   : run_mxfp4_gemm(wrows, arows);
        if (g.nr != nr || g.nc != nc) { std::printf("  setup failed nr=%d nc=%d\n", nr, nc); ++failures; continue; }
        ++cases; ++fcases;

        const int depth = accumulation_depth(fmt, n);
        for (int m = 0; m < nr; ++m)
        for (int j = 0; j < nc; ++j) {
            const size_t k = (size_t)m*nc + j;
            const double mag = g.intermediate_mag[k];
            const Verdict v_gg = check(g.simd[k], g.gemv_rows[k], mag, depth*256, 256);
            const Verdict v_sg = check(g.simd[k], g.generic[k],   mag, depth*256, 256);
            const Verdict v_rf = check(g.simd[k], g.reference[k], mag, depth*256, 256);
            if (v_gg.ratio > w_gg) w_gg = v_gg.ratio;
            if (v_sg.ratio > w_sg) w_sg = v_sg.ratio;
            if (v_rf.ratio > w_rf) w_rf = v_rf.ratio;
            if (v_gg.ratio > f_gg) f_gg = v_gg.ratio;
            if (v_sg.ratio > f_sg) f_sg = v_sg.ratio;
            if (v_rf.ratio > f_rf) f_rf = v_rf.ratio;
            if (!v_gg.ok || !v_sg.ok || !v_rf.ok) {
                if (++failures <= 8) {
                    std::printf("!! %s %s nr=%d nc=%d nb=%d seed=%d [m=%d,j=%d]\n",
                                format_name(fmt), input_class_name(cls), nr, nc, nblocks, seed, m, j);
                    std::printf("   gemm=%.9g gemv=%.9g generic=%.9g ref=%.9g\n",
                                (double)g.simd[k], (double)g.gemv_rows[k],
                                (double)g.generic[k], g.reference[k]);
                    std::printf("   ratios: gemm-vs-gemv=%.4g simd-vs-generic=%.4g simd-vs-ref=%.4g\n",
                                v_gg.ratio, v_sg.ratio, v_rf.ratio);
                }
            }
        }
    }

    std::printf("  %-6s %5lld cases  gemm-vs-gemv=%.6f  simd-vs-generic=%.6f  simd-vs-ref=%.6f\n",
                format_name(fmt), fcases, f_gg, f_sg, f_rf);
  }

    std::printf("\ntotal %lld gemm cases, worst: gemm-vs-gemv=%.6f simd-vs-generic=%.6f simd-vs-ref=%.6f\n",
                cases, w_gg, w_sg, w_rf);
    std::printf(failures ? "\nGEMM FAILED (%d)\n" : "\nGEMM CLEAN\n", failures);
    return failures ? 1 : 0;
}
