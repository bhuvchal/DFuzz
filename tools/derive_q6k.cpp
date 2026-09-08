// Hand-derivation input for the q6_K finding.
//
// Dumps the exact quantized state the kernels operate on: every nonzero
// dequantized weight and activation, the per-group scales, the integer
// products, and the exact partial sums. From these the admissible error can
// be computed by hand rather than asserted.
#include "dfuzz/reference.h"
#include "dfuzz/init.h"
#include "dfuzz/buildinfo.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
using namespace dfuzz;

extern "C" {
void quantize_row_q6_K_ref(const float*, void*, int64_t);
void quantize_row_q8_K_ref(const float*, void*, int64_t);
void dequantize_row_q6_K(const void*, float*, int64_t);
void ggml_vec_dot_q6_K_q8_K        (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_q6_K_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
}

struct blk6 { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
struct blk8 { float d; int8_t qs[256]; int16_t bsums[16]; };

static double fp16(uint16_t h){
    const uint32_t s=(h>>15)&1u,e=(h>>10)&0x1Fu,m=h&0x3FFu;
    double v;
    if(!e) v=(double)m*5.9604644775390625e-08;
    else if(e==0x1F) v=m?__builtin_nan(""):__builtin_inf();
    else v=(1.0+(double)m/1024.0)*__builtin_ldexp(1.0,(int)e-15);
    return s?-v:v;
}

int main(){
    dfuzz::Guard g; print_build_config();

    const int N=256;
    std::vector<float> x(N,0.f), y(N,0.f);
    x[87]=11429230.0f; x[140]=-774724.25f; x[143]=510633.031f;
    y[86]=-7282.96387f; y[143]=65.4250183f;

    std::vector<uint8_t> xq(210), yq(292);
    quantize_row_q6_K_ref(x.data(), xq.data(), N);
    quantize_row_q8_K_ref(y.data(), yq.data(), N);
    const blk6* b6=reinterpret_cast<const blk6*>(xq.data());
    const blk8* b8=reinterpret_cast<const blk8*>(yq.data());

    std::printf("\n# quantized state\n");
    std::printf("  q6_K super-block scale d      = %.17g  (fp16 bits 0x%04x)\n", fp16(b6->d), b6->d);
    std::printf("  q8_K super-block scale d8     = %.17g\n", (double)b8->d);
    std::printf("  q6_K group scales (16 groups of 16 elements):\n   ");
    for(int i=0;i<16;++i) std::printf(" %4d", (int)b6->scales[i]);
    std::printf("\n");

    std::vector<float> xd(N);
    dequantize_row_q6_K(xq.data(), xd.data(), N);

    std::printf("\n# nonzero elements after quantization\n");
    std::printf("  %-5s %-6s %16s %16s %20s\n","idx","group","x_dequant","y = d8*q8","product");
    double exact=0.0, abs_sum=0.0;
    for(int i=0;i<N;++i){
        const double xv=(double)xd[i];
        const double yv=(double)b8->d*(double)b8->qs[i];
        const double p=xv*yv;
        exact+=p; abs_sum+=std::fabs(p);
        if(xv!=0.0 || yv!=0.0)
            std::printf("  %-5d %-6d %16.9g %16.9g %20.10g\n", i, i/16, xv, yv, p);
    }
    std::printf("\n  exact dot (fp64)      = %.17g\n", exact);
    std::printf("  sum|x_i y_i|          = %.17g\n", abs_sum);

    // Per-group integer partial sums: what the kernel accumulates exactly.
    std::printf("\n# per-group integer sums (kernel accumulates these in int32, exactly)\n");
    std::printf("  %-6s %8s %12s %20s\n","group","scale","sum(q6*q8)","contribution");
    double sum_contrib=0.0, max_partial=0.0;
    const double d6=fp16(b6->d), d8=(double)b8->d;
    for(int gi=0; gi<16; ++gi){
        long long si=0;
        for(int k=gi*16;k<(gi+1)*16;++k){
            // recover the 6-bit signed quant from the dequantized value
            const double sc = d6*(double)b6->scales[gi];
            const long long q6 = (sc!=0.0)? (long long)std::llround((double)xd[k]/sc) : 0;
            si += q6*(long long)b8->qs[k];
        }
        const double contrib = d6*(double)b6->scales[gi]*d8*(double)si;
        sum_contrib += contrib;
        max_partial = std::max(max_partial, std::fabs(contrib));
        if(si!=0) std::printf("  %-6d %8d %12lld %20.10g\n", gi, (int)b6->scales[gi], si, contrib);
    }
    std::printf("\n  sum of group contributions = %.17g\n", sum_contrib);
    std::printf("  largest |group contribution| = %.17g\n", max_partial);

    float simd=0, gen=0;
    ggml_vec_dot_q6_K_q8_K        (N,&simd,0,xq.data(),0,yq.data(),0,1);
    ggml_vec_dot_q6_K_q8_K_generic(N,&gen ,0,xq.data(),0,yq.data(),0,1);
    std::printf("\n# kernel results on this host\n");
    std::printf("  simd    = %.9g   abs err = %.6g\n",(double)simd,std::fabs((double)simd-exact));
    std::printf("  generic = %.9g   abs err = %.6g\n",(double)gen, std::fabs((double)gen -exact));
    std::printf("  ulp(result) in fp32 = %.6g\n",
        (double)std::nextafterf((float)exact, 3e38f) - (double)(float)exact);
    return 0;
}
