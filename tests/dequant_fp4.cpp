// Bit-exact validation of the fp4-format references against ggml.
// Same discipline as dequant_check: exactness, not closeness.
#include "dfuzz/formats.h"
#include "dfuzz/buildinfo.h"
#include "dfuzz/init.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

extern "C" {
void quantize_row_mxfp4_ref(const float*, void*, int64_t);
void quantize_row_nvfp4_ref(const float*, void*, int64_t);
void quantize_row_q8_0_ref (const float*, void*, int64_t);
void dequantize_row_mxfp4(const void*, float*, int64_t);
void dequantize_row_nvfp4(const void*, float*, int64_t);
void dequantize_row_q8_0 (const void*, float*, int64_t);
}

using namespace dfuzz;

static int check(const char* label, const std::vector<float>& src, int nblocks,
                 void (*quant)(const float*, void*, int64_t),
                 void (*ggml_deq)(const void*, float*, int64_t),
                 void (*our_deq)(const void*, int, std::vector<double>&),
                 int bytes, int elems) {
    const int n = nblocks * elems;
    std::vector<uint8_t> q((size_t)nblocks * bytes);
    quant(src.data(), q.data(), n);
    std::vector<float> theirs(n);
    ggml_deq(q.data(), theirs.data(), n);
    std::vector<double> ours;
    our_deq(q.data(), nblocks, ours);
    int bad = 0;
    for (int i = 0; i < n && (int)ours.size() == n; ++i) {
        const float f = (float)ours[i];
        uint32_t a, b; std::memcpy(&a,&f,4); std::memcpy(&b,&theirs[i],4);
        if (a != b) {
            if (++bad <= 3)
                std::printf("  %-6s [%3d] ours=%.9g (0x%08x) theirs=%.9g (0x%08x)\n",
                            label, i, (double)f, a, (double)theirs[i], b);
        }
    }
    if ((int)ours.size() != n) { std::printf("  %-6s FAIL size\n", label); return 1; }
    std::printf("  %-6s %s: %d/%d bit-identical\n", label, bad?"FAIL":"ok", n-bad, n);
    return bad ? 1 : 0;
}

int main() {
    dfuzz::Guard guard;   // ggml lookup tables must be live before any kernel call
    print_build_config();
    std::printf("\n# fp4-format dequantization validation\n");
    const char* kinds[] = {"normal","allzero","mixedmag","pow2","onebig","denormal","tiny"};
    int failures = 0;
    std::mt19937 rng(0xBEEF);
    for (const char* k : kinds) {
        // 4 blocks of the largest format; sized per format below.
        auto gen = [&](int n) {
            std::vector<float> v(n);
            if (!strcmp(k,"normal"))        { std::normal_distribution<float> d(0,1); for(auto&e:v)e=d(rng); }
            else if (!strcmp(k,"allzero"))  { std::fill(v.begin(),v.end(),0.f); }
            else if (!strcmp(k,"mixedmag")) { std::uniform_int_distribution<int> e(-30,30); std::normal_distribution<float> m(0,1); for(auto&x:v)x=std::ldexp(m(rng),e(rng)); }
            else if (!strcmp(k,"pow2"))     { std::uniform_int_distribution<int> e(-8,8),s(0,1); for(auto&x:v)x=std::ldexp(s(rng)?1.f:-1.f,e(rng)); }
            else if (!strcmp(k,"onebig"))   { std::normal_distribution<float> d(0,1); for(auto&e:v)e=d(rng); v[0]=1e30f; }
            else if (!strcmp(k,"denormal")) { std::uniform_int_distribution<int> s(0,1); for(auto&x:v)x=(s(rng)?1.f:-1.f)*1e-42f; }
            else if (!strcmp(k,"tiny"))     { std::uniform_int_distribution<int> s(0,1); for(auto&x:v)x=(s(rng)?1.f:-1.f)*1e-30f; }
            return v;
        };
        std::printf("input=%s\n", k);
        failures += check("mxfp4", gen(4*QK_MXFP4), 4, quantize_row_mxfp4_ref, dequantize_row_mxfp4, dequantize_mxfp4, block_bytes(Format::mxfp4), QK_MXFP4);
        failures += check("nvfp4", gen(4*QK_NVFP4), 4, quantize_row_nvfp4_ref, dequantize_row_nvfp4, dequantize_nvfp4, block_bytes(Format::nvfp4), QK_NVFP4);
        failures += check("q8_0",  gen(4*QK8_0),    4, quantize_row_q8_0_ref,  dequantize_row_q8_0,  dequantize_q8_0,  34, QK8_0);
    }
    std::printf(failures ? "\nFP4 DEQUANT FAILED (%d)\n" : "\nFP4 DEQUANT PASSED\n", failures);
    return failures ? 1 : 0;
}
