#pragma once
#include "dfuzz/case.h"
#include "dfuzz/oracle.h"
#include <concepts>
#include <cstdio>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

namespace dfuzz {

// A reduction pass proposes simpler candidates derived from a case. The loop
// keeps any candidate that still exhibits the same failure.
//
// The concept is what makes the loop generic: any type satisfying it can be
// dropped into shrink() with no change to the loop, and a malformed pass is a
// compile error at the call site rather than a runtime surprise.
template <typename P>
concept ReductionPass = requires(P p, const Case& c, std::size_t i) {
    // Human-readable, for the shrink log.
    { p.name() } -> std::convertible_to<std::string_view>;
    // How many candidates this pass offers for this case. May be 0.
    { p.candidate_count(c) } -> std::convertible_to<std::size_t>;
    // Candidate i, or nullopt if it would be illegal (e.g. not block-aligned).
    { p.candidate(c, i) } -> std::same_as<std::optional<Case>>;
};

// --- The three passes -----------------------------------------------------

// 1. Halve the length. Highest-value move: every success halves the work for
//    every later pass. Constrained to whole blocks, since the kernels read
//    fixed-size blocks and a partial one would read past the buffer.
//    Also offers near-sizes, not only halves: a bug in the tail handling of a
//    partial block disappears at conveniently-sized inputs, so pure halving
//    can shrink a real bug out of existence.
class HalvePass {
public:
    std::string_view name() const { return "halve"; }
    std::size_t candidate_count(const Case& c) const;
    std::optional<Case> candidate(const Case& c, std::size_t i) const;
};

// 2. Zero out a contiguous chunk. Chunks first (halves, quarters, ...), so
//    large irrelevant regions vanish in one step instead of element by element.
class ZeroChunkPass {
public:
    std::string_view name() const { return "zero"; }
    std::size_t candidate_count(const Case& c) const;
    std::optional<Case> candidate(const Case& c, std::size_t i) const;
};

// 3. Replace a surviving value with a simple, readable one. Changes nothing
//    about severity and everything about legibility: [1.0, 1.0, 1e-8] tells a
//    story that [-0.4471637, 1.2093344, 0.0002216] does not.
class SimplifyValuePass {
public:
    std::string_view name() const { return "simplify"; }
    std::size_t candidate_count(const Case& c) const;
    std::optional<Case> candidate(const Case& c, std::size_t i) const;
};

// --- The fixpoint loop ----------------------------------------------------

struct ShrinkStats {
    int       rounds       = 0;
    int       accepted     = 0;
    long long evaluations  = 0;
    int       n_before     = 0;
    int       n_after      = 0;
    int       nonzero_after = 0;
};

struct ShrinkOptions {
    bool verbose        = false;
    long long max_evals = 200000;   // safety valve
};

// Runs passes to exhaustion: sweep all passes, and if any accepted a
// reduction, sweep again -- a successful cut often unlocks cuts that failed
// before. Stops when a full sweep changes nothing (a local minimum: every
// reduction the passes know how to make would fix the bug).
//
// Variadic over passes so the pass set is a compile-time choice; each pack
// element is checked against ReductionPass at instantiation.
template <ReductionPass... Passes>
ShrinkStats shrink(Case& c, const Signature& want, Oracle& oracle,
                   const ShrinkOptions& opt, Passes&... passes);

// Non-template entry point using the standard three-pass set.
ShrinkStats shrink_default(Case& c, const Signature& want, Oracle& oracle,
                           const ShrinkOptions& opt = {});

// --- template implementation ---------------------------------------------
namespace detail {

template <ReductionPass P>
bool run_pass(P& p, Case& c, const Signature& want, Oracle& oracle,
              const ShrinkOptions& opt, ShrinkStats& st) {
    bool progress = false;
    // Re-query the count each iteration: accepting a candidate changes the
    // case, and with it how many candidates remain meaningful.
    for (std::size_t i = 0; i < p.candidate_count(c); ++i) {
        if (oracle.evaluations() >= opt.max_evals) break;
        auto cand = p.candidate(c, i);
        if (!cand) continue;
        if (oracle.still_fails(*cand, want)) {
            if (opt.verbose) {
                std::printf("  %-9.*s n=%4d -> %4d  kept\n",
                            (int)p.name().size(), p.name().data(), c.n(), cand->n());
            }
            c = std::move(*cand);   // move, not copy: candidates own big vectors
            ++st.accepted;
            progress = true;
            i = (std::size_t)-1;    // restart this pass on the reduced case
        }
    }
    return progress;
}

} // namespace detail

template <ReductionPass... Passes>
ShrinkStats shrink(Case& c, const Signature& want, Oracle& oracle,
                   const ShrinkOptions& opt, Passes&... passes) {
    ShrinkStats st;
    st.n_before = c.n();
    const long long ev0 = oracle.evaluations();

    bool progress = true;
    while (progress && oracle.evaluations() < opt.max_evals) {
        ++st.rounds;
        progress = false;
        // Fold over the pack: every pass runs each round, in declaration order.
        ((progress |= detail::run_pass(passes, c, want, oracle, opt, st)), ...);
    }

    st.evaluations = oracle.evaluations() - ev0;
    st.n_after = c.n();
    for (float v : c.x) if (v != 0.0f) ++st.nonzero_after;
    return st;
}

} // namespace dfuzz
