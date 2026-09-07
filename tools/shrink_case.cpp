// Shrink a specific failing case to a minimal reproducer.
//
// Must run on the platform where the divergence occurs -- the x86 q6_K
// divergence does not reproduce on arm64 at all, so shrinking it locally
// would find nothing.
//
// Prints the minimized case as literal values so the reproducer stands alone,
// runnable against llama.cpp with no part of this harness involved.
#include "dfuzz/generate.h"
#include "dfuzz/oracle.h"
#include "dfuzz/shrink.h"
#include "dfuzz/init.h"
#include "dfuzz/buildinfo.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

using namespace dfuzz;

static Format parse_fmt(const char* s) {
    if (!std::strcmp(s, "q4_K"))  return Format::q4_K;
    if (!std::strcmp(s, "q6_K"))  return Format::q6_K;
    if (!std::strcmp(s, "nvfp4")) return Format::nvfp4;
    return Format::mxfp4;
}
static InputClass parse_cls(const char* s) {
    for (int i = 0; i < (int)InputClass::Count; ++i)
        if (!std::strcmp(s, input_class_name((InputClass)i))) return (InputClass)i;
    return InputClass::Uniform;
}

int main(int argc, char** argv) {
    dfuzz::Guard guard;
    print_build_config();

    const char* fmt_s = "q6_K";
    const char* cls_s = "mixed-magnitude";
    int blocks = 1;
    uint64_t seed = 0x9e3579b87e4a7c19ull;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--fmt")    && i+1 < argc) fmt_s  = argv[++i];
        else if (!std::strcmp(argv[i], "--class")  && i+1 < argc) cls_s  = argv[++i];
        else if (!std::strcmp(argv[i], "--blocks") && i+1 < argc) blocks = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed")   && i+1 < argc) seed   = std::strtoull(argv[++i], nullptr, 0);
    }

    GenSpec sp;
    sp.fmt = parse_fmt(fmt_s);
    sp.cls = parse_cls(cls_s);
    sp.blocks = blocks;
    sp.seed = seed;

    Case c = generate(sp);
    Oracle oracle;
    const Outcome o = oracle.evaluate(c);

    std::printf("\n# shrink target: %s/%s blocks=%d seed=0x%llx\n",
                format_name(sp.fmt), input_class_name(sp.cls), blocks,
                (unsigned long long)seed);
    std::printf("  simd=%.9g generic=%.9g reference=%.17g\n",
                o.simd, o.generic, o.reference);
    std::printf("  worst axis=%s ratio=%.4f  failed=%d\n",
                axis_name(o.worst), o.ratio, (int)o.failed);
    std::printf("  |simd-ref|/|ref|    = %.4e\n",
                std::fabs(o.simd - o.reference) / std::fabs(o.reference));
    std::printf("  |generic-ref|/|ref| = %.4e\n",
                std::fabs(o.generic - o.reference) / std::fabs(o.reference));

    if (!o.failed) {
        std::printf("\nCase does not fail on this platform -- nothing to shrink.\n");
        return 0;
    }

    ShrinkOptions opt;
    opt.verbose = true;
    opt.max_evals = 200000;
    const ShrinkStats st = shrink_default(c, o.sig, oracle, opt);

    const Outcome f = oracle.evaluate(c);
    std::printf("\n# minimized: n %d -> %d (%d nonzero x), %d rounds, %lld evals\n",
                st.n_before, st.n_after, st.nonzero_after, st.rounds, st.evaluations);
    std::printf("  simd=%.9g generic=%.9g reference=%.17g ratio=%.4f\n",
                f.simd, f.generic, f.reference, f.ratio);
    std::printf("  |simd-ref|/|ref|    = %.4e\n",
                std::fabs(f.simd - f.reference) / std::fabs(f.reference));
    std::printf("  |generic-ref|/|ref| = %.4e\n",
                std::fabs(f.generic - f.reference) / std::fabs(f.reference));

    // Sparse form first: after shrinking almost everything is zero, and an
    // index/value list is what actually goes into a bug report.
    std::printf("\n# nonzero elements (index = value)\n");
    std::printf("  x:");
    for (int i = 0; i < c.n(); ++i)
        if (c.x[i] != 0.0f) std::printf("  x[%d]=%.9gf;", i, (double)c.x[i]);
    std::printf("\n  y:");
    for (int i = 0; i < c.n(); ++i)
        if (c.y[i] != 0.0f) std::printf("  y[%d]=%.9gf;", i, (double)c.y[i]);
    std::printf("\n");

    // Literal values, so the reproducer is self-contained.
    std::printf("\n# reproducer (%s, n=%d)\n", format_name(c.fmt), c.n());
    std::printf("static const float x[%d] = {", c.n());
    for (int i = 0; i < c.n(); ++i)
        std::printf("%s%.9g%s", i % 8 == 0 ? "\n  " : " ", (double)c.x[i],
                    i + 1 < c.n() ? "," : "");
    std::printf("\n};\nstatic const float y[%d] = {", c.n());
    for (int i = 0; i < c.n(); ++i)
        std::printf("%s%.9g%s", i % 8 == 0 ? "\n  " : " ", (double)c.y[i],
                    i + 1 < c.n() ? "," : "");
    std::printf("\n};\n");
    return 0;
}
