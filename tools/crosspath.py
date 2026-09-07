#!/usr/bin/env python3
"""Cross-path comparison: do two builds' kernels agree with each other?

Each build emits one TSV of kernel results for an identical, seed-reproducible
case set. This compares them pairwise. A divergence here involves no reference
of ours at all: the same input, the same format, two different SIMD paths,
different answers.

Rounding differences are expected -- the paths accumulate in different orders.
What matters is the SIZE, judged against the same per-format depth model the
C++ harness uses, so a flagged row is a genuine bound violation rather than a
different ULP.
"""
import sys
from collections import defaultdict

U_F32 = 5.9604644775390625e-08  # 2^-24

DEPTH_PER_ELEM = {          # fp32 ops whose order can differ, per element
    "mxfp4": 1.0 / 32,      # one fp32 add per 32-element block
    "nvfp4": 1.0 / 16,      # one per 16-element sub-block
    "q4_K":  16.0 / 256,    # 16 per 256-element super-block
    "q6_K":  16.0 / 256,
}
BLOCK_ELEMS = {"mxfp4": 32, "nvfp4": 64, "q4_K": 256, "q6_K": 256}


def gamma(k):
    ku = k * U_F32
    return 1e300 if ku >= 1.0 else ku / (1.0 - ku)


def load(path):
    rows = {}
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(header):
                continue
            key = (f[idx["fmt"]], f[idx["class"]], f[idx["blocks"]], f[idx["seed"]])
            rows[key] = (float(f[idx["simd"]]),
                         float(f[idx["generic"]]),
                         float(f[idx["reference"]]))
    return rows


def compare(name_a, a, name_b, b):
    shared = a.keys() & b.keys()
    if not shared:
        print(f"  {name_a} vs {name_b}: NO SHARED CASES "
              f"({len(a)} vs {len(b)} rows) -- case sets differ, cannot compare")
        return 1

    exact = 0
    within = 0
    violations = []
    for key in sorted(shared):
        fmt, cls, blocks, seed = key
        sa, _, ref = a[key]
        sb, _, _ = b[key]
        if sa == sb:
            exact += 1
            continue
        n = int(blocks) * BLOCK_ELEMS[fmt]
        depth = max(1.0, DEPTH_PER_ELEM[fmt] * n)
        # Bound against the reference magnitude; the emitted rows do not carry
        # abs_sum, so this is deliberately conservative and will only flag
        # differences far larger than accumulation order can explain.
        allowed = gamma(depth) * abs(ref) * 2.0 + U_F32 * abs(ref)
        diff = abs(sa - sb)
        if diff <= allowed:
            within += 1
        else:
            violations.append((key, sa, sb, diff, allowed, diff / allowed if allowed else float("inf")))

    print(f"  {name_a} vs {name_b}: {len(shared)} cases, "
          f"{exact} bit-identical, {within} within bound, {len(violations)} VIOLATIONS")
    for (fmt, cls, blocks, seed), sa, sb, diff, allowed, ratio in violations[:10]:
        print(f"    !! {fmt}/{cls} blocks={blocks} seed={seed}")
        print(f"       {name_a}={sa!r}  {name_b}={sb!r}")
        print(f"       diff={diff:.6g} allowed={allowed:.6g} ratio={ratio:.2f}")
    if len(violations) > 10:
        print(f"    ... and {len(violations) - 10} more")
    return len(violations)


def main(argv):
    if len(argv) < 3:
        print("usage: crosspath.py NAME=FILE [NAME=FILE ...]", file=sys.stderr)
        return 2
    builds = []
    for arg in argv[1:]:
        name, _, path = arg.partition("=")
        builds.append((name, load(path)))
        print(f"loaded {name}: {len(builds[-1][1])} rows")

    print("\n# pairwise cross-path comparison")
    total = 0
    for i in range(len(builds)):
        for j in range(i + 1, len(builds)):
            total += compare(builds[i][0], builds[i][1], builds[j][0], builds[j][1])

    print()
    if total:
        print(f"CROSS-PATH DIVERGENCE: {total} violations")
    else:
        print("CROSS-PATH CLEAN: all builds agree within the derived bound")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
