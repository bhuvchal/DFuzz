// x86 reported q6_K/mixed-magnitude cases where the SIMD kernel returns a
// nonzero value while both the generic kernel and the fp64 reference return
// exactly zero. That cannot be explained by accumulation order.
//
// Replays the exact seeds and dumps enough state to tell a kernel defect from
// an out-of-domain input.
#include "dfuzz/generate.h"
#include "dfuzz/oracle.h"
#include "dfuzz/reference.h"
#include "dfuzz/init.h"
#include "dfuzz/buildinfo.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
using namespace dfuzz;
extern "C" {
void quantize_row_q6_K_ref(const float*,void*,int64_t);
void quantize_row_q8_K_ref(const float*,void*,int64_t);
void ggml_vec_dot_q6_K_q8_K        (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_q6_K_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
}
int main(){
  dfuzz::Guard g; print_build_config();
  const uint64_t seeds[] = {
    0x9e3579b87e4a7c17ull, 0x9e3579b87e4a7c11ull,
    0x9e3579b87e4a7c1dull, 0x9e3579b87e4a7c1bull,
  };
  std::printf("\n# q6_K/mixed-magnitude replay, blocks=1\n");
  for (uint64_t sd : seeds) {
    GenSpec sp; sp.fmt=Format::q6_K; sp.cls=InputClass::MixedMagnitude; sp.blocks=1; sp.seed=sd;
    Case c = generate(sp);
    const int n = c.n();
    std::vector<uint8_t> xq(210), yq(292);
    quantize_row_q6_K_ref(c.x.data(), xq.data(), n);
    quantize_row_q8_K_ref(c.y.data(), yq.data(), n);
    float simd=0, gen=0;
    ggml_vec_dot_q6_K_q8_K        (n,&simd,0,xq.data(),0,yq.data(),0,1);
    ggml_vec_dot_q6_K_q8_K_generic(n,&gen ,0,xq.data(),0,yq.data(),0,1);
    std::vector<double> xd, yd;
    dequantize_q6_K(xq.data(),1,xd); dequantize_q8_K(yq.data(),1,yd);
    RefResult rr = reference_dot_blocked(xd, yd, 32);

    // Block state: q6_K d is fp16; q8_K d is fp32.
    uint16_t xd16; std::memcpy(&xd16, xq.data()+208, 2);
    float y_d; std::memcpy(&y_d, yq.data(), 4);
    float maxx=0, maxy=0;
    for(float v:c.x) maxx=std::max(maxx,std::fabs(v));
    for(float v:c.y) maxy=std::max(maxy,std::fabs(v));
    int nonzero_x=0; for(double v:xd) if(v!=0.0) ++nonzero_x;
    int nonzero_y=0; for(double v:yd) if(v!=0.0) ++nonzero_y;

    std::printf("\nseed=0x%llx\n", (unsigned long long)sd);
    std::printf("  simd=%.9g generic=%.9g ref=%.17g abs_sum=%.6g\n",
                (double)simd,(double)gen,rr.dot,rr.abs_sum);
    std::printf("  q6_K d(fp16 bits)=0x%04x -> %g   q8_K d=%g\n",
                xd16, (double)__builtin_ldexp(1.0,0)*0 + (double)(xd16?1:0), (double)y_d);
    std::printf("  max|x_in|=%g max|y_in|=%g   dequant nonzero: x=%d/%d y=%d/%d\n",
                (double)maxx,(double)maxy,nonzero_x,(int)xd.size(),nonzero_y,(int)yd.size());
    if (simd != gen)
      std::printf("  *** SIMD != GENERIC by %.6g ***\n", std::fabs((double)simd-(double)gen));
  }
  return 0;
}
