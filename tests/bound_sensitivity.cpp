// A bound that never fires is worthless. This asserts the predicate still
// catches faults, by perturbing a correct kernel result by known amounts and
// checking where detection begins.
#include "dfuzz/generate.h"
#include "dfuzz/oracle.h"
#include "dfuzz/init.h"
#include "dfuzz/buildinfo.h"
#include <cstdio>
#include <cmath>

using namespace dfuzz;

int main() {
    dfuzz::Guard guard;
    print_build_config();
    std::printf("\n# bound sensitivity: smallest detected perturbation\n");
    std::printf("%-8s %-16s %14s %14s\n", "format", "class", "rel-perturb", "detected");

    Oracle o;
    int failures = 0;
    for (Format fmt : {Format::mxfp4, Format::nvfp4, Format::q4_K, Format::q6_K}) {
        for (InputClass cls : {InputClass::Uniform, InputClass::MixedMagnitude}) {
            GenSpec sp; sp.fmt = fmt; sp.cls = cls; sp.blocks = 8; sp.seed = 12345;
            const Case c = generate(sp);
            const Outcome base = o.evaluate(c);
            if (base.failed) { std::printf("  base case already fails; skipping\n"); continue; }

            // Find the smallest relative perturbation of the SIMD result that
            // the predicate rejects. A correct bound detects something well
            // below 1e-3 relative on these sizes.
            double detected = 0.0;
            for (double rel = 1e-9; rel <= 1e-1; rel *= 2.0) {
                const double perturbed = base.simd * (1.0 + rel);
                const int depth = accumulation_depth(fmt, c.n());
                const Verdict v = check(perturbed, base.reference,
                                        base.abs_sum, depth * 256, 256);
                if (!v.ok) { detected = rel; break; }
            }
            std::printf("%-8s %-16s %14.3g %14s\n", format_name(fmt),
                        input_class_name(cls), detected,
                        detected > 0 ? "yes" : "NEVER");
            if (detected == 0.0 || detected > 1e-2) {
                std::printf("   FAIL: bound is too loose to be useful here\n");
                ++failures;
            }
        }
    }
    std::printf(failures ? "\nSENSITIVITY FAILED (%d)\n" : "\nSENSITIVITY OK\n", failures);
    return failures ? 1 : 0;
}
