// Is the gemm-vs-gemv gap explained by the ACTIVATION packing?
//
// gemm consumes ggml_quantize_mat_q8_K_4x8 output; gemv consumes
// quantize_row_q8_K_ref. If those two disagree about the quantized values or
// the per-row scale, the kernels are legitimately computing different things
// and there is no kernel bug to report.
//
// Compares them directly: same float rows in, are the int8 quants and scales
// identical?
#include "dfuzz/generate.h"
#include "dfuzz/init.h"
#include "dfuzz/buildinfo.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
using namespace dfuzz;
extern "C" {
void quantize_row_q8_K_ref(const float*,void*,int64_t);
void ggml_quantize_mat_q8_K_4x8(const float*, void*, int64_t);
}
struct b8K  { float d; int8_t qs[256]; int16_t bsums[16]; };

int main(){
  dfuzz::Guard g; print_build_config();
  const int nb=5, n=nb*256, nr=8;
  std::vector<std::vector<float>> a(nr);
  for(int m=0;m<nr;++m){
    GenSpec sp; sp.fmt=Format::q4_K; sp.cls=InputClass::Uniform; sp.blocks=nb;
    sp.seed = 0xBEEF00ull ^ ((uint64_t)InputClass::Uniform<<36) ^ ((uint64_t)nr<<24)
            ^ ((uint64_t)nb<<12) ^ ((uint64_t)2<<4) ^ (uint64_t)m;
    a[m]=generate(sp).y;
  }
  std::vector<float> flat((size_t)nr*n);
  for(int m=0;m<nr;++m) std::memcpy(flat.data()+(size_t)m*n, a[m].data(), (size_t)n*sizeof(float));

  const size_t q8Kx4 = 4*sizeof(float)+4*256+(256/4)*sizeof(int16_t);
  std::vector<uint8_t> packed((size_t)(nr/4)*nb*q8Kx4);
  for(int gi=0;gi<nr/4;++gi)
    ggml_quantize_mat_q8_K_4x8(flat.data()+(size_t)gi*4*n, packed.data()+(size_t)gi*nb*q8Kx4, n);

  std::printf("\n# activation packing: mat_q8_K_4x8 vs row_q8_K_ref\n");
  for(int m=0;m<nr;++m){
    std::vector<uint8_t> rowbuf((size_t)nb*292);
    quantize_row_q8_K_ref(a[m].data(), rowbuf.data(), n);
    const int grp=m/4, lane=m%4;
    int dmismatch=0; double dmax=0;
    for(int b=0;b<nb;++b){
      const uint8_t* pk = packed.data()+(size_t)grp*nb*q8Kx4+(size_t)b*q8Kx4;
      float dp; std::memcpy(&dp, pk+lane*sizeof(float), sizeof(float));
      const b8K* rw = reinterpret_cast<const b8K*>(rowbuf.data()+(size_t)b*292);
      if(dp!=rw->d){ ++dmismatch; dmax=std::max(dmax,(double)std::fabs(dp-rw->d)/std::fabs(rw->d)); }
    }
    std::printf("  m=%d (grp=%d lane=%d): scale mismatches %d/%d, max rel %.3e\n",
                m, grp, lane, dmismatch, nb, dmax);
  }
  return 0;
}
