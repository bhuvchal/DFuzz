#include "dfuzz/shrink.h"
#include <algorithm>
#include <cmath>

namespace dfuzz {

// --- HalvePass ------------------------------------------------------------
// Candidates, in order of aggressiveness:
//   0 : half the blocks
//   1 : three quarters
//   2 : one block fewer   <- the "near size" that catches tail-handling bugs
//   3 : one block          (the floor)
std::size_t HalvePass::candidate_count(const Case& c) const {
    // 4 truncations (keep a prefix) + 4 tail slices (keep a suffix).
    // Prefix-only shrinking can never reduce a bug that lives in the last
    // block: every truncation destroys it and is correctly rejected, so the
    // pass silently does nothing. Offering suffixes fixes that blind spot.
    return c.blocks() > 1 ? 8u : 0u;
}

std::optional<Case> HalvePass::candidate(const Case& c, std::size_t i) const {
    const int be = block_elems(c.fmt);
    const int nb = c.blocks();
    const bool suffix = i >= 4;
    int want_nb;
    switch (i % 4) {
        case 0: want_nb = nb / 2;         break;
        case 1: want_nb = (nb * 3) / 4;   break;
        case 2: want_nb = nb - 1;         break;
        case 3: want_nb = 1;              break;
        default: return std::nullopt;
    }
    if (want_nb < 1 || want_nb >= nb) return std::nullopt;

    Case out = c;
    const int n = want_nb * be;
    if (suffix) {
        // Keep the LAST want_nb blocks.
        const int drop = c.n() - n;
        out.x.assign(c.x.begin() + drop, c.x.end());
        out.y.assign(c.y.begin() + drop, c.y.end());
    } else {
        out.x.resize(n);
        out.y.resize(n);
    }
    return out;
}

// --- ZeroChunkPass --------------------------------------------------------
// Chunk sizes descend by halves; within each size, every aligned chunk of the
// x operand then of the y operand. Coarse-to-fine means large irrelevant
// regions disappear in one evaluation rather than n of them.
namespace {
struct ChunkPlan { int len; int off; bool on_y; };

ChunkPlan plan_chunk(const Case& c, std::size_t i) {
    const int n = c.n();
    int len = n;
    std::size_t base = 0;
    while (len >= 1) {
        const std::size_t count = (std::size_t)((n + len - 1) / len) * 2u; // x and y
        if (i < base + count) {
            const std::size_t local = i - base;
            const int slot = (int)(local / 2);
            const bool on_y = (local % 2) == 1;
            return ChunkPlan{len, slot * len, on_y};
        }
        base += count;
        len /= 2;
    }
    return ChunkPlan{0, 0, false};
}
} // namespace

std::size_t ZeroChunkPass::candidate_count(const Case& c) const {
    const int n = c.n();
    if (n <= 0) return 0;
    std::size_t total = 0;
    for (int len = n; len >= 1; len /= 2)
        total += (std::size_t)((n + len - 1) / len) * 2u;
    return total;
}

std::optional<Case> ZeroChunkPass::candidate(const Case& c, std::size_t i) const {
    const ChunkPlan p = plan_chunk(c, i);
    if (p.len <= 0 || p.off >= c.n()) return std::nullopt;

    auto& src = p.on_y ? c.y : c.x;
    const int hi = std::min(p.off + p.len, (int)src.size());
    // Skip if already all zero -- re-testing a no-op wastes an evaluation.
    bool any = false;
    for (int k = p.off; k < hi; ++k) if (src[k] != 0.0f) { any = true; break; }
    if (!any) return std::nullopt;

    Case out = c;
    auto& dst = p.on_y ? out.y : out.x;
    for (int k = p.off; k < hi; ++k) dst[k] = 0.0f;
    return out;
}

// --- SimplifyValuePass ----------------------------------------------------
// A short ladder of readable constants, tried in order of simplicity. The
// pass only proposes a replacement that is strictly simpler than what is
// there, so it terminates.
namespace {
const float kLadder[] = {1.0f, -1.0f, 0.5f, 2.0f, 1e-8f, -1e-8f};
constexpr std::size_t kLadderN = sizeof(kLadder) / sizeof(kLadder[0]);

int simplicity_rank(float v) {
    for (std::size_t k = 0; k < kLadderN; ++k) if (v == kLadder[k]) return (int)k;
    return 1000; // anything not on the ladder is "complex"
}
} // namespace

std::size_t SimplifyValuePass::candidate_count(const Case& c) const {
    return (std::size_t)c.n() * 2u * kLadderN;
}

std::optional<Case> SimplifyValuePass::candidate(const Case& c, std::size_t i) const {
    const std::size_t per = 2u * kLadderN;
    const int idx = (int)(i / per);
    const std::size_t rem = i % per;
    const bool on_y = (rem / kLadderN) == 1;
    const std::size_t lad = rem % kLadderN;
    if (idx >= c.n()) return std::nullopt;

    auto& src = on_y ? c.y : c.x;
    const float cur = src[idx];
    if (cur == 0.0f) return std::nullopt;                  // already simplest
    if (simplicity_rank(cur) <= (int)lad) return std::nullopt; // not an improvement

    Case out = c;
    (on_y ? out.y : out.x)[idx] = kLadder[lad];
    return out;
}

// --- default driver -------------------------------------------------------
ShrinkStats shrink_default(Case& c, const Signature& want, Oracle& oracle,
                           const ShrinkOptions& opt) {
    HalvePass         halve;
    ZeroChunkPass     zero;
    SimplifyValuePass simplify;
    return shrink(c, want, oracle, opt, halve, zero, simplify);
}

} // namespace dfuzz
