// The repacked (interleaved) weight surface: mxfp4 4x4 gemv.
//
// Strongest axis here is gemv-vs-vecdot: the repacked kernel must reproduce,
// row for row, what the plain dot product computes on the same weights. That
// makes the interleaving falsifiable with no reference of ours involved.
#include "dfuzz/gemv.h"
#include "dfuzz/generate.h"
#include "dfuzz/bound.h"
#include "dfuzz/init.h"
#include "dfuzz/buildinfo.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace dfuzz;

int main() {
    dfuzz::Guard guard;
    print_build_config();
    std::printf("\n# repacked-weight surface: mxfp4 4x4 and q4_K 8x8 gemv\n");

    const InputClass classes[] = {
        InputClass::Uniform, InputClass::MixedMagnitude, InputClass::PowerOfTwo,
        InputClass::Cancellation, InputClass::Denormal, InputClass::SignedZero,
        InputClass::Saturating, InputClass::AlternatingMax, InputClass::SingleSpike,
    };
    const int col_counts_mx[] = {4, 8, 12, 16, 32};
    const int col_counts_q4[] = {8, 16, 24, 32};
    const int block_counts[] = {1, 2, 3, 5, 8, 17};

    int failures = 0;
    long long cases = 0;
    double worst_gv = 0.0, worst_sg = 0.0, worst_ref = 0.0;

  for (int fmt_i = 0; fmt_i < 2; ++fmt_i) {
    const bool is_q4 = (fmt_i == 1);
    const Format fmt = is_q4 ? Format::q4_K : Format::mxfp4;
    const int be = block_elems(fmt);
    const int ncols_n = is_q4 ? 4 : 5;
    const int* col_counts = is_q4 ? col_counts_q4 : col_counts_mx;
    double w_gv = 0, w_sg = 0, w_rf = 0; long long fcases = 0;

    for (InputClass cls : classes) {
        for (int ci = 0; ci < ncols_n; ++ci) {
            const int nc = col_counts[ci];
            for (int nblocks : block_counts) {
                for (int seed = 0; seed < 8; ++seed) {
                    const int n = nblocks * be;
                    // Build nc independent weight rows plus one activation row,
                    // each from the biased generator.
                    std::vector<std::vector<float>> rows((size_t)nc);
                    for (int j = 0; j < nc; ++j) {
                        GenSpec sp; sp.fmt = fmt; sp.cls = cls;
                        sp.blocks = nblocks;
                        sp.seed = 0xABCDEFull ^ ((uint64_t)cls << 40) ^ ((uint64_t)nc << 28)
                                ^ ((uint64_t)nblocks << 16) ^ ((uint64_t)seed << 8) ^ (uint64_t)j;
                        rows[(size_t)j] = generate(sp).x;
                    }
                    GenSpec ap; ap.fmt = fmt; ap.cls = cls; ap.blocks = nblocks;
                    ap.seed = 0x1234567ull ^ ((uint64_t)cls << 32) ^ ((uint64_t)seed << 4);
                    const std::vector<float> act = generate(ap).y;

                    const GemvResult g = is_q4 ? run_q4_K_gemv(rows, act)
                                               : run_mxfp4_gemv(rows, act);
                    if ((int)g.simd.size() != nc) { std::printf("  setup failed\n"); ++failures; continue; }
                    ++cases; ++fcases;

                    for (int j = 0; j < nc; ++j) {
                        const int depth = accumulation_depth(fmt, n);
                        const double mag = g.intermediate_mag[(size_t)j];

                        // The decisive axis: repacked kernel vs plain dot product.
                        const Verdict v_gv = check(g.simd[j], g.vecdot[j], mag, depth*256, 256);
                        const Verdict v_sg = check(g.simd[j], g.generic[j], mag, depth*256, 256);
                        const Verdict v_rf = check(g.simd[j], g.reference[j], mag, depth*256, 256);
                        if (v_gv.ratio > worst_gv) worst_gv = v_gv.ratio;
                        if (v_sg.ratio > worst_sg) worst_sg = v_sg.ratio;
                        if (v_rf.ratio > worst_ref) worst_ref = v_rf.ratio;
                        if (v_gv.ratio > w_gv) w_gv = v_gv.ratio;
                        if (v_sg.ratio > w_sg) w_sg = v_sg.ratio;
                        if (v_rf.ratio > w_rf) w_rf = v_rf.ratio;

                        if (!v_gv.ok || !v_sg.ok || !v_rf.ok) {
                            if (++failures <= 6) {
                                std::printf("!! %s %s nc=%d nb=%d seed=%d row=%d\n",
                                            format_name(fmt), input_class_name(cls), nc, nblocks, seed, j);
                                std::printf("   gemv=%.9g vecdot=%.9g generic=%.9g ref=%.9g\n",
                                            (double)g.simd[j], (double)g.vecdot[j],
                                            (double)g.generic[j], g.reference[j]);
                                std::printf("   ratios: gemv-vs-vecdot=%.4g simd-vs-generic=%.4g simd-vs-ref=%.4g\n",
                                            v_gv.ratio, v_sg.ratio, v_rf.ratio);
                            }
                        }
                    }
                }
            }
        }
    }

    std::printf("  %-6s %5lld cases  gemv-vs-vecdot=%.6f  simd-vs-generic=%.6f  simd-vs-ref=%.6f\n",
                format_name(fmt), fcases, w_gv, w_sg, w_rf);
  }

    std::printf("\ntotal %lld gemv cases, worst: gemv-vs-vecdot=%.6f simd-vs-generic=%.6f simd-vs-ref=%.6f\n",
                cases, worst_gv, worst_sg, worst_ref);
    std::printf(failures ? "\nGEMV FAILED (%d)\n" : "\nGEMV CLEAN\n", failures);
    return failures ? 1 : 0;
}
