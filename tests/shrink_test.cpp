// Validates the shrinker against a bug whose minimal form we already know.
//
// A real divergence is not available on demand, so we inject one: a fake
// oracle reports "failed" exactly when a planted marker survives in the case.
// Because we know the marker, we know what the minimal case must look like,
// and can assert the shrinker actually reaches it rather than merely
// producing something smaller.
#include "dfuzz/shrink.h"
#include "dfuzz/init.h"
#include "dfuzz/buildinfo.h"
#include <cstdio>
#include <random>

using namespace dfuzz;

namespace {

// Stand-in for Oracle with the same interface the passes rely on. The planted
// bug: "fails iff x[k] is a large negative value for some k in the last block,
// AND at least 3 nonzero values remain in that block".
struct FakeOracle {
    long long evals = 0;
    int  target_block_elems = 32;
    bool still_fails(const Case& c, const Signature&) {
        ++evals;
        const int be = target_block_elems;
        if (c.n() < be) return false;
        const int start = c.n() - be;
        int nonzero = 0; bool marker = false;
        for (int i = start; i < c.n(); ++i) {
            if (c.x[i] != 0.0f) ++nonzero;
            if (c.x[i] <= -100.0f) marker = true;
        }
        return marker && nonzero >= 3;
    }
    long long evaluations() const { return evals; }
};

// shrink() is written against Oracle&; give the fake the same shape by
// templating a local copy of the loop. Reuses the real passes and the real
// fixpoint logic -- only the oracle is substituted.
template <ReductionPass... Passes>
ShrinkStats shrink_with(FakeOracle& o, Case& c, const Signature& want,
                        const ShrinkOptions& opt, Passes&... passes) {
    ShrinkStats st;
    st.n_before = c.n();
    const long long ev0 = o.evaluations();
    bool progress = true;
    while (progress && o.evaluations() < opt.max_evals) {
        ++st.rounds;
        progress = false;
        auto run = [&](auto& p) {
            bool made = false;
            for (std::size_t i = 0; i < p.candidate_count(c); ++i) {
                if (o.evaluations() >= opt.max_evals) break;
                auto cand = p.candidate(c, i);
                if (!cand) continue;
                if (o.still_fails(*cand, want)) {
                    if (opt.verbose)
                        std::printf("  %-9s n=%4d -> %4d  kept\n",
                                    std::string(p.name()).c_str(), c.n(), cand->n());
                    c = std::move(*cand);
                    ++st.accepted; made = true;
                    i = (std::size_t)-1;
                }
            }
            return made;
        };
        ((progress |= run(passes)), ...);
    }
    st.evaluations = o.evaluations() - ev0;
    st.n_after = c.n();
    for (float v : c.x) if (v != 0.0f) ++st.nonzero_after;
    return st;
}

} // namespace

int main() {
    dfuzz::Guard guard;
    print_build_config();
    std::printf("\n# shrinker validation (planted bug, known minimal form)\n");

    // A large random case with the marker planted in the final block.
    Case c;
    c.fmt = Format::mxfp4;
    const int n = 4096;
    c.x.resize(n); c.y.resize(n);
    std::mt19937 rng(99);
    std::normal_distribution<float> d(0.f, 1.f);
    for (int i = 0; i < n; ++i) { c.x[i] = d(rng); c.y[i] = d(rng); }
    c.x[n - 5] = -1234.5f;   // the marker

    FakeOracle fake;
    Signature want{};
    ShrinkOptions opt; opt.verbose = true; opt.max_evals = 200000;

    HalvePass halve; ZeroChunkPass zero; SimplifyValuePass simplify;
    std::printf("start: n=%d nonzero=%d\n", c.n(), n);
    const ShrinkStats st = shrink_with(fake, c, want, opt, halve, zero, simplify);

    std::printf("\nresult: n %d -> %d, nonzero=%d, rounds=%d, accepted=%d, evals=%lld\n",
                st.n_before, st.n_after, st.nonzero_after, st.rounds, st.accepted, st.evaluations);
    std::printf("surviving x values:");
    for (int i = 0; i < c.n(); ++i) if (c.x[i] != 0.0f) std::printf(" [%d]=%g", i, (double)c.x[i]);
    std::printf("\n");

    // The known minimal form: one block (32), exactly 3 nonzero, marker present.
    int fails = 0;
    if (st.n_after != 32)      { std::printf("FAIL: expected n=32, got %d\n", st.n_after); ++fails; }
    if (st.nonzero_after != 3) { std::printf("FAIL: expected 3 nonzero, got %d\n", st.nonzero_after); ++fails; }
    bool marker = false;
    for (float v : c.x) if (v <= -100.0f) marker = true;
    if (!marker) { std::printf("FAIL: marker lost -- shrinker walked into a different bug\n"); ++fails; }
    if (!fake.still_fails(c, want)) { std::printf("FAIL: final case does not reproduce\n"); ++fails; }

    std::printf(fails ? "\nSHRINK VALIDATION FAILED (%d)\n" : "\nSHRINK VALIDATION PASSED\n", fails);
    return fails ? 1 : 0;
}
