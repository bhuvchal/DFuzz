// Diagnostic for the x86 avx2 gemm divergence.
//
// Question: is the disagreement between gemm and gemv caused by our packing,
// or by the kernels? Isolates it by feeding the SAME repacked weights to both
// and reporting where the paths part company, plus which q8 packing each side
// consumed.
#include "dfuzz/gemv.h"
#include "dfuzz/generate.h"
#include "dfuzz/init.h"
#include "dfuzz/buildinfo.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
using namespace dfuzz;

extern "C" {
void quantize_row_q4_K_ref(const float*,void*,int64_t);
void quantize_row_q8_K_ref(const float*,void*,int64_t);
void ggml_quantize_mat_q8_K_4x8(const float*, void*, int64_t);
void ggml_gemv_q4_K_8x8_q8_K        (int,float*,size_t,const void*,const void*,int,int);
void ggml_gemm_q4_K_8x8_q8_K        (int,float*,size_t,const void*,const void*,int,int);
void ggml_gemm_q4_K_8x8_q8_K_generic(int,float*,size_t,const void*,const void*,int,int);
}

int main(){
  dfuzz::Guard g;
  print_build_config();
  std::printf("\n# gemm vs gemv diagnostic (same packed weights)\n");

  const int nc=8, nb=5, n=nb*256;
  for (int nr : {4, 8}) {
    std::vector<std::vector<float>> w(nc), a(nr);
    for(int j=0;j<nc;++j){GenSpec s;s.fmt=Format::q4_K;s.cls=InputClass::Uniform;s.blocks=nb;s.seed=200+j;w[j]=generate(s).x;}
    for(int m=0;m<nr;++m){GenSpec s;s.fmt=Format::q4_K;s.cls=InputClass::Uniform;s.blocks=nb;s.seed=800+m;a[m]=generate(s).y;}

    std::vector<uint8_t> rowq((size_t)nc*nb*144);
    for(int j=0;j<nc;++j) quantize_row_q4_K_ref(w[j].data(), rowq.data()+(size_t)j*nb*144, n);
    std::vector<uint8_t> packed; repack_q4_K_8x8(rowq.data(), nc, nb, packed);

    std::vector<float> aflat((size_t)nr*n);
    for(int m=0;m<nr;++m) std::memcpy(aflat.data()+(size_t)m*n, a[m].data(), (size_t)n*sizeof(float));
    const size_t q8Kx4 = 4*sizeof(float)+4*256+(256/4)*sizeof(int16_t);
    std::vector<uint8_t> aq((size_t)(nr/4)*nb*q8Kx4);
    for(int gi=0; gi<nr/4; ++gi)
      ggml_quantize_mat_q8_K_4x8(aflat.data()+(size_t)gi*4*n, aq.data()+(size_t)gi*nb*q8Kx4, n);

    std::vector<float> gm((size_t)nr*nc,0), gmg((size_t)nr*nc,0);
    ggml_gemm_q4_K_8x8_q8_K        (n, gm.data(),  (size_t)nc, packed.data(), aq.data(), nr, nc);
    ggml_gemm_q4_K_8x8_q8_K_generic(n, gmg.data(), (size_t)nc, packed.data(), aq.data(), nr, nc);

    std::printf("\nnr=%d\n", nr);
    for(int m=0;m<nr;++m){
      std::vector<uint8_t> yq((size_t)nb*292);
      quantize_row_q8_K_ref(a[m].data(), yq.data(), n);
      std::vector<float> gv(nc,0);
      ggml_gemv_q4_K_8x8_q8_K(n, gv.data(), 0, packed.data(), yq.data(), 1, nc);
      double worst_gg=0, worst_gv=0;
      for(int j=0;j<nc;++j){
        size_t k=(size_t)m*nc+j;
        double mag=std::fabs((double)gv[j]);
        worst_gg=std::max(worst_gg, std::fabs((double)gm[k]-(double)gmg[k])/(mag>0?mag:1));
        worst_gv=std::max(worst_gv, std::fabs((double)gm[k]-(double)gv[j])/(mag>0?mag:1));
      }
      std::printf("  m=%d  gemm-simd vs gemm-generic: %.2e   gemm vs gemv: %.2e %s\n",
                  m, worst_gg, worst_gv, worst_gv>1e-4?"  <-- LARGE":"");
    }
  }
  return 0;
}
