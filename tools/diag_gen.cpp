// Is generation actually portable? Prints a fingerprint of the raw engine, the
// portable primitives, and the finished cases. Two platforms printing the same
// lines means a seed denotes the same input; anything else means the
// cross-path comparison is comparing different problems.
#include "dfuzz/generate.h"
#include "dfuzz/buildinfo.h"
#include "dfuzz/init.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
using namespace dfuzz;

static uint64_t fnv(const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

int main(){
  dfuzz::Guard g; print_build_config();
  std::printf("\n# generation fingerprint\n");

  std::mt19937_64 rng(0x9e3779b97f4a7c15ull);
  std::printf("  engine[0..3]: %llu %llu %llu %llu\n",
    (unsigned long long)rng(),(unsigned long long)rng(),
    (unsigned long long)rng(),(unsigned long long)rng());

  std::printf("  ldexp(1.0,-30)=%.17g  log(0.5)=%.17g  cos(1.0)=%.17g  sqrt(2.0)=%.17g\n",
    std::ldexp(1.0,-30), std::log(0.5), std::cos(1.0), std::sqrt(2.0));

  for (int ci = 0; ci < (int)InputClass::Count; ++ci) {
    GenSpec sp; sp.fmt=Format::q6_K; sp.cls=(InputClass)ci; sp.blocks=1;
    sp.seed = 0x9e3579b87e4a7c24ull;
    Case c = generate(sp);
    float mx=0; for(float v:c.x) mx=std::max(mx,std::fabs(v));
    std::printf("  %-16s x[0]=%.9g x[1]=%.9g max|x|=%.6g hash=%016llx\n",
      input_class_name((InputClass)ci), (double)c.x[0], (double)c.x[1], (double)mx,
      (unsigned long long)fnv(c.x.data(), c.x.size()*sizeof(float)));
  }
  return 0;
}
