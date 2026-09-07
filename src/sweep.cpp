// The sweep: cross product of format x input class x shape, every finding
// shrunk and reported with the seed that produced it.
#include "dfuzz/generate.h"
#include "dfuzz/oracle.h"
#include "dfuzz/shrink.h"
#include "dfuzz/init.h"
#include "dfuzz/buildinfo.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace dfuzz;

namespace {

struct Finding {
    GenSpec   spec;
    Outcome   outcome;
    Case      minimal;
    ShrinkStats shrink;
};

// Block counts chosen to sit on and beside the seams: 1 block, a couple, an
// odd count, a power of two, and one that is prime so no blocking strategy
// divides it evenly.
const int kBlockCounts[] = {1, 2, 3, 5, 7, 8, 16, 17, 31, 64};

struct Tally {
    long long cases = 0;
    double    worst_ratio = 0.0;
    GenSpec   worst_spec{};
};

} // namespace

int main(int argc, char** argv) {
    dfuzz::Guard guard;

    long long seeds_per_cell = 32;
    bool verbose_shrink = false;
    const char* emit_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--seeds") && i + 1 < argc) seeds_per_cell = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--verbose")) verbose_shrink = true;
        // Emit every kernel result as data so two builds -- AVX2 vs AVX-512 vs
        // NEON -- can be compared against EACH OTHER, not merely each against
        // the reference. That cross-path axis is unreachable from a single
        // machine and is the main reason this project has CI.
        else if (!std::strcmp(argv[i], "--emit") && i + 1 < argc) emit_path = argv[++i];
    }

    std::FILE* emit = emit_path ? std::fopen(emit_path, "w") : nullptr;
    if (emit_path && !emit) {
        std::fprintf(stderr, "cannot open %s for writing\n", emit_path);
        return 2;
    }
    if (emit) std::fprintf(emit, "fmt\tclass\tblocks\tseed\tsimd\tgeneric\treference\n");

    print_build_config();
    std::printf("#   seeds/cell    : %lld\n", seeds_per_cell);

    const Format formats[] = {Format::mxfp4, Format::nvfp4, Format::q4_K, Format::q6_K};

    Oracle oracle;
    std::vector<Finding> findings;
    std::map<std::string, Tally> by_cell;   // "fmt/class" -> tally
    long long total = 0;

    for (Format fmt : formats) {
        for (int ci = 0; ci < (int)InputClass::Count; ++ci) {
            const InputClass cls = (InputClass)ci;
            const std::string cell = std::string(format_name(fmt)) + "/" + input_class_name(cls);
            for (int blocks : kBlockCounts) {
                for (long long s = 0; s < seeds_per_cell; ++s) {
                    GenSpec spec;
                    spec.fmt    = fmt;
                    spec.cls    = cls;
                    spec.blocks = blocks;
                    // Seed mixes every axis so a cell's seeds are independent
                    // across shapes, and the whole run is reproducible.
                    spec.seed = ((uint64_t)ci << 48) ^ ((uint64_t)blocks << 32)
                              ^ ((uint64_t)fmt << 24) ^ (uint64_t)s ^ 0x9E3779B97F4A7C15ull;

                    const Case c = generate(spec);
                    const Outcome o = oracle.evaluate(c);
                    ++total;

                    if (emit) {
                        // %.9g round-trips fp32 exactly, so a byte-identical
                        // line means a byte-identical kernel result.
                        std::fprintf(emit, "%s\t%s\t%d\t0x%llx\t%.9g\t%.9g\t%.17g\n",
                                     format_name(fmt), input_class_name(cls), blocks,
                                     (unsigned long long)spec.seed,
                                     o.simd, o.generic, o.reference);
                    }

                    Tally& t = by_cell[cell];
                    ++t.cases;
                    if (o.ratio > t.worst_ratio) { t.worst_ratio = o.ratio; t.worst_spec = spec; }

                    if (o.failed) {
                        Case minimal = c;
                        ShrinkOptions opt; opt.verbose = verbose_shrink; opt.max_evals = 50000;
                        const ShrinkStats st = shrink_default(minimal, o.sig, oracle, opt);
                        findings.push_back(Finding{spec, o, std::move(minimal), st});
                        std::printf("\n!! FINDING  %s/%s blocks=%d seed=0x%llx\n",
                                    format_name(fmt), input_class_name(cls), blocks,
                                    (unsigned long long)spec.seed);
                        std::printf("   axis=%s ratio=%.4g  simd=%.9g generic=%.9g ref=%.9g\n",
                                    axis_name(o.worst), o.ratio, o.simd, o.generic, o.reference);
                        std::printf("   shrunk n %d -> %d (%d nonzero), %lld evals\n",
                                    st.n_before, st.n_after, st.nonzero_after, st.evaluations);
                    }
                }
            }
        }
    }

    // --- coverage claim ----------------------------------------------------
    // This table IS the deliverable when no bugs surface: across this cross
    // product, conformance holds within the derived bound, and here is how
    // close each cell came to the bound.
    std::printf("\n# coverage: %lld cases, %zu findings\n", total, findings.size());
    std::printf("%-26s %8s %12s %s\n", "cell", "cases", "worst-ratio", "headroom");
    std::vector<std::pair<std::string, Tally>> rows(by_cell.begin(), by_cell.end());
    std::sort(rows.begin(), rows.end(),
              [](auto& a, auto& b){ return a.second.worst_ratio > b.second.worst_ratio; });
    for (auto& [name, t] : rows) {
        std::printf("%-26s %8lld %12.6f %8.1fx\n", name.c_str(), t.cases, t.worst_ratio,
                    t.worst_ratio > 0 ? 1.0 / t.worst_ratio : 0.0);
    }

    std::printf("\n# oracle: %lld evaluations, %lld buffer growths\n",
                oracle.evaluations(), oracle.allocations());
    if (emit) { std::fclose(emit); std::printf("\n# results emitted to %s\n", emit_path); }

    std::printf(findings.empty()
        ? "\nSWEEP CLEAN: conformance holds across all cells within the derived bound\n"
        : "\nSWEEP FOUND %zu DIVERGENCES\n", findings.size());
    return findings.empty() ? 0 : 1;
}
