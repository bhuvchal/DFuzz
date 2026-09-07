#include "dfuzz/buildinfo.h"
#include <cstdio>

extern "C" {
// ggml exposes runtime feature detection; we report what the dispatcher
// could pick, so a result record says which SIMD path actually ran.
int ggml_cpu_has_neon(void);
int ggml_cpu_has_dotprod(void);
int ggml_cpu_has_matmul_int8(void);
int ggml_cpu_has_sve(void);
int ggml_cpu_has_avx2(void);
int ggml_cpu_has_avx512(void);
int ggml_cpu_has_f16c(void);
}

namespace dfuzz {

void print_build_config() {
    std::printf("# build config\n");

#if defined(__clang__)
    std::printf("#   compiler      : clang %d.%d.%d\n", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    std::printf("#   compiler      : gcc %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    std::printf("#   compiler      : unknown\n");
#endif

#if defined(__OPTIMIZE__)
    std::printf("#   optimization  : on (-O1 or higher)\n");
#else
    std::printf("#   optimization  : off (-O0)\n");
#endif

    // -ffast-math must be OFF. It permits reassociation and flushes denormals,
    // which would silently invalidate every bound in this harness.
#if defined(__FAST_MATH__)
    std::printf("#   fast-math     : ON  *** INVALID: bounds assume IEEE semantics ***\n");
#else
    std::printf("#   fast-math     : off (IEEE semantics)\n");
#endif
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
    std::printf("#   finite-math   : ON  *** INVALID: NaN/Inf handling disabled ***\n");
#endif

    std::printf("#   host arch     : %s\n",
#if defined(__aarch64__) || defined(_M_ARM64)
        "arm64"
#elif defined(__x86_64__) || defined(_M_X64)
        "x86_64"
#else
        "other"
#endif
    );

    // Which kernel the dispatcher will select. On arm64 only the NEON family
    // is reachable, so cross-path comparison (AVX2 vs AVX-512 vs NEON) is
    // unavailable locally and must run in x86 CI.
    std::printf("#   cpu features  : neon=%d dotprod=%d i8mm=%d sve=%d avx2=%d avx512=%d f16c=%d\n",
                ggml_cpu_has_neon(), ggml_cpu_has_dotprod(), ggml_cpu_has_matmul_int8(),
                ggml_cpu_has_sve(), ggml_cpu_has_avx2(), ggml_cpu_has_avx512(),
                ggml_cpu_has_f16c());
    // Which repacked kernels actually have a distinct SIMD path here. On x86
    // ggml aliases mxfp4 gemv/gemm _generic onto the plain name, so that
    // comparison axis does not exist -- the coverage claim must say so.
    std::printf("#   repack simd   : mxfp4=%s q4_K=%s\n",
#if defined(__aarch64__) || defined(_M_ARM64)
                "yes", "yes"
#else
                "no (generic is the implementation)", "yes"
#endif
    );
    std::fflush(stdout);
}

} // namespace dfuzz
