// The x86 AVX2 q6_K kernel consumes y[i].bsums (precomputed sums of qs in
// groups of 16) to apply its offset correction cheaply. Our reference, and
// the generic kernel, work from qs directly.
//
// If bsums disagrees with the actual sum of qs, the two are computing
// genuinely different things -- and since bsums is int16, the obvious failure
// mode is OVERFLOW: 16 values of magnitude up to 127 sum to at most 2032, but
// quantize_row_q8_K_ref clamps to +-127 per element, so the bound holds...
// unless the encoder produces values outside that range.
#include "dfuzz/generate.h"
#include "dfuzz/init.h"
#include "dfuzz/buildinfo.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
using namespace dfuzz;
extern "C" { void quantize_row_q8_K_ref(const float*,void*,int64_t); }
struct b8K { float d; int8_t qs[256]; int16_t bsums[16]; };

int main(){
  dfuzz::Guard g; print_build_config();
  const uint64_t seeds[]={0x9e3579b87e4a7c17ull,0x9e3579b87e4a7c12ull,
                          0x9e3579b87e4a7c19ull,0x9e3579b87e4a7c1bull,0x9e3579b87e4a7c33ull};
  std::printf("\n# q8_K bsums vs recomputed sums (q6_K/mixed-magnitude)\n");
  for(uint64_t sd:seeds){
    GenSpec sp;sp.fmt=Format::q6_K;sp.cls=InputClass::MixedMagnitude;sp.blocks=1;sp.seed=sd;
    Case c=generate(sp);
    std::vector<uint8_t> yq(292);
    quantize_row_q8_K_ref(c.y.data(), yq.data(), 256);
    const b8K* b = reinterpret_cast<const b8K*>(yq.data());
    int mismatches=0, worst=0, qmin=127, qmax=-128;
    for(int j=0;j<16;++j){
      int sum=0;
      for(int k=0;k<16;++k){ int v=b->qs[j*16+k]; sum+=v; qmin=std::min(qmin,v); qmax=std::max(qmax,v); }
      if(sum != b->bsums[j]){ ++mismatches; worst=std::max(worst,std::abs(sum-(int)b->bsums[j])); }
    }
    std::printf("  0x%llx: bsums mismatches=%d worst=%d  qs range [%d,%d] %s\n",
      (unsigned long long)sd, mismatches, worst, qmin, qmax,
      mismatches? "  <-- DISAGREE":"");
  }
  return 0;
}
