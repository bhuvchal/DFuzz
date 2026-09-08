# Derivation: the x86 q6_K result is outside any admissible fp32 bound

The shrinker reduced the case far enough that the error analysis becomes
exact rather than statistical. This settles the question left open in
FINDING.md.

## The reduced computation

After minimization the 256-element super-block has exactly one contributing
group. Confirmed identical on arm64 and x86 (`tools/derive_q6k.cpp` run on
both):

| quantity | value | representation |
|---|---|---|
| `d` (q6_K super-block scale) | `2790` | fp16 `0x6973`, exact |
| `scales[8]` (group scale) | `9` | int8, exact |
| `d8` (q8_K scale) | `57.346172332763672` | fp32, exact by definition |
| `sum(q6*q8)` for group 8 | `20` | int32, **exact — no rounding** |

Every other group contributes zero. The kernel's entire task is

```
result = d * scales[8] * d8 * sumi
       = 2790 * 9 * 57.346172332763672 * 20
```

with exact value

```
28 799 247.745513916
```

**There is no accumulation.** One group, one integer sum. The usual
argument — different summation order gives a different answer — does not
apply, because there is nothing to reorder.

## The admissible interval

All four factors are exactly representable in fp32, so error can only enter
through rounding of the products. At most three multiplies are needed, and
IEEE-754 requires each to be correctly rounded (error <= 0.5 ulp). The
standard bound is therefore

```
|error| <= ((1+u)^3 - 1) * |result|,   u = 2^-24
         = 5.1497
         = 2.575 ulp        (1 ulp = 2.0 at this magnitude)
```

That is the worst case over *any* evaluation order. Enumerating all 24
orderings of the four factors in fp32 gives only three distinct results:

| result | error | ulp | orderings |
|---|---|---|---|
| `28799246.0` | 1.745514 | 0.873 | 4 |
| `28799248.0` | 0.254486 | 0.127 | 16 |
| `28799250.0` | 2.254486 | 1.127 | 4 |

So the true admissible set is **1.127 ulp**, comfortably inside the 2.575 ulp
bound.

## The observation

| | result | bits | error |
|---|---|---|---|
| nearest fp32 to exact | `28799248.0` | `0x4bdbb888` | 0.127 ulp |
| generic (x86 and arm64) | `28799250.0` | `0x4bdbb889` | **1.127 ulp** |
| arm64 NEON | `28799250.0` | `0x4bdbb889` | **1.127 ulp** |
| **x86 SIMD** | `28794880.0` | `0x4bdbb000` | **2183.9 ulp** |

The generic and NEON results sit exactly on the worst admissible ordering.
The x86 SIMD result is **2185 representable floats away** from them, and
exceeds the theoretical bound by **848x**.

## Conclusion

No correctly-rounded fp32 evaluation of `2790 * 9 * 57.346172 * 20`, in any
order, can produce `28794880.0`. The value is outside the admissible interval
by nearly three orders of magnitude.

This is not a tolerance question, and the three benign explanations are all
excluded by construction:

- **Accumulation order** — there is only one term; nothing to reorder.
- **FMA contraction** — contraction removes roundings, it cannot move a result
  848x outside the bound. Confirmed empirically: identical under
  `-ffp-contract=off`.
- **Quantization difference** — the dumped state is byte-identical on both
  platforms.

## Note on the mechanism

The bit pattern is suggestive: x86's low 11 mantissa bits are zero, and
`28794880 / 4096 = 7030` exactly against a true `7031.066`. That is the
signature of a value that passed through a reduced-precision intermediate.

Three candidate mechanisms were tested and rejected:

1. `_mm256_maddubs_epi16` saturation — worst-case pair sum is 16002, well
   inside int16 range.
2. An integer offset error in the `q8sclsub` bias correction — the implied
   discrepancy is fractional (0.0273), not integral.
3. Pre-multiplying the scales into one fp32 `d` before scaling by `sumi` —
   modelling this reproduces `28799250.0`, the correct result.

The mechanism is therefore **not yet identified**. The finding does not depend
on it: the observed value is outside the admissible interval regardless of how
it is produced. Identifying the mechanism is upstream's to do, and a bug
report should say so rather than guess.
