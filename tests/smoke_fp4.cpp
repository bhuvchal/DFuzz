#include "dfuzz/formats.h"
#include "dfuzz/reference.h"
#include "dfuzz/bound.h"
#include "dfuzz/buildinfo.h"
#include "dfuzz/init.h"
#include <cstdio>
#include <random>
#include <vector>

extern "C" {
void quantize_row_mxfp4_ref(const float*, void*, int64_t);
void quantize_row_nvfp4_ref(const float*, void*, int64_t);
void quantize_row_q8_0_ref (const float*, void*, int64_t);
void ggml_vec_dot_mxfp4_q8_0        (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_mxfp4_q8_0_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_nvfp4_q8_0        (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_nvfp4_q8_0_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
}

using namespace dfuzz;

int main() {
    dfuzz::Guard guard;   // ggml lookup tables must be live before any kernel call
    print_build_config();
    std::printf("\n# fp4 kernels vs fp64 reference (depth is per-format)\n");
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    int failures = 0;

    for (int which = 0; which < 2; ++which) {
        const Format fmt = which ? Format::nvfp4 : Format::mxfp4;
        const int elems  = block_elems(fmt);
        const int nb     = 16;
        const int n      = nb * elems;

        std::vector<float> xf(n), yf(n);
        for (int i = 0; i < n; ++i) { xf[i] = nd(rng); yf[i] = nd(rng); }

        std::vector<uint8_t> xq((size_t)nb * block_bytes(fmt));
        std::vector<uint8_t> yq((size_t)(n / QK8_0) * 34);
        (which ? quantize_row_nvfp4_ref : quantize_row_mxfp4_ref)(xf.data(), xq.data(), n);
        quantize_row_q8_0_ref(yf.data(), yq.data(), n);

        float s_simd = 0, s_gen = 0;
        if (which) {
            ggml_vec_dot_nvfp4_q8_0        (n,&s_simd,0,xq.data(),0,yq.data(),0,1);
            ggml_vec_dot_nvfp4_q8_0_generic(n,&s_gen ,0,xq.data(),0,yq.data(),0,1);
        } else {
            ggml_vec_dot_mxfp4_q8_0        (n,&s_simd,0,xq.data(),0,yq.data(),0,1);
            ggml_vec_dot_mxfp4_q8_0_generic(n,&s_gen ,0,xq.data(),0,yq.data(),0,1);
        }

        std::vector<double> xd, yd;
        (which ? dequantize_nvfp4 : dequantize_mxfp4)(xq.data(), nb, xd);
        dequantize_q8_0(yq.data(), n / QK8_0, yd);
        const RefResult ref = reference_dot(xd, yd);

        const int depth = accumulation_depth(fmt, n);
        std::printf("%s  n=%d  depth=%d (%s)\n", format_name(fmt), n, depth,
                    which ? "one fp32 add per 16-elem sub-block"
                          : "one fp32 add per 32-elem block, E8M0 scale exact");
        std::printf("   simd=%.9g generic=%.9g ref=%.9g  abs_sum=%.6g\n",
                    (double)s_simd, (double)s_gen, ref.dot, ref.abs_sum);

        struct { const char* nm; double a, b; } cs[] = {
            {"simd    vs ref    ", (double)s_simd, ref.dot},
            {"generic vs ref    ", (double)s_gen,  ref.dot},
            {"simd    vs generic", (double)s_simd, (double)s_gen},
        };
        for (auto& c : cs) {
            // depth-aware bound: pass the per-format effective n
            const Verdict v = check(c.a, c.b, ref.abs_sum, depth * 256, 256);
            std::printf("   %s obs=%.4g allowed=%.4g ratio=%.4f %s\n",
                        c.nm, v.observed, v.allowed, v.ratio, v.ok?"OK":"FAIL");
            if (!v.ok) ++failures;
        }
    }
    std::printf(failures ? "\nFP4 SMOKE FAILED (%d)\n" : "\nFP4 SMOKE PASSED\n", failures);
    return failures ? 1 : 0;
}
