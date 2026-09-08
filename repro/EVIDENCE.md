# Evidence sheet — x86 q6_K vec_dot

Raw facts for writing the upstream issue. Not a draft: no prose to reuse.
Every number below is reproduced by `repro/q6k_x86.c`.

## Versions
- llama.cpp commit: `d03efa5d5369a9cba63bec1ae712a2f6e686a1d6` (2026-09-06)
- Failing: GCC 13.3.0, x86_64, ubuntu-24.04 (GitHub Actions runner)
- Passing: Apple clang 17.0.0, arm64 (M4)
- Runner CPU flags: avx2=1 avx512=1 f16c=1

## Affected symbol
`ggml_vec_dot_q6_K_q8_K`, x86 AVX2 path
(`ggml/src/ggml-cpu/arch/x86/quants.c`, `#if defined __AVX2__` branch)

## Input (n = 256, one q6_K super-block; all other elements 0.0f)
```
x[87]  =  11429230.0f
x[140] = -774724.25f
x[143] =  510633.031f
y[86]  = -7282.96387f
y[143] =  65.4250183f
```

## Quantized state (byte-identical on both platforms; verified in CI)
| quantity | value |
|---|---|
| q6_K super-block scale `d` | 2790 (fp16 bits `0x6973`) |
| q8_K scale `d8` | 57.346172332763672 |
| `scales[8]` | 9 |
| `sum(q6*q8)` group 8 | 20 (int32, exact) |
| all other groups | contribute 0 |

Task reduces to: `2790 * 9 * 57.346172332763672 * 20`

## Numbers
| | value | fp32 bits | error |
|---|---|---|---|
| exact (fp64) | 28799247.745513916 | — | — |
| nearest fp32 | 28799248.0 | `0x4bdbb888` | 0.127 ulp |
| generic, both platforms | 28799250.0 | `0x4bdbb889` | 1.127 ulp |
| arm64 NEON | 28799250.0 | `0x4bdbb889` | 1.127 ulp |
| x86 AVX2 | 28794880.0 | `0x4bdbb000` | 2183.9 ulp |

- 1 ulp at this magnitude = 2.0
- x86 and generic are 2185 representable floats apart
- `simd - generic = -4370`

## Bound
- All four factors exactly representable in fp32; <=3 multiplies
- IEEE-754: each correctly rounded, error <= 0.5 ulp
- `((1+u)^3 - 1) * result = 5.1497` absolute = **2.575 ulp**, u = 2^-24
- All 24 evaluation orders enumerated: only 3 distinct fp32 results, worst
  **1.127 ulp**
- x86 exceeds the theoretical bound by **848x**

## Excluded explanations
| explanation | why excluded |
|---|---|
| accumulation order | one contributing group; nothing to reorder |
| FMA contraction | identical under `-ffp-contract=off`; contraction removes roundings |
| quantization difference | dumped state byte-identical both platforms |
| harness artifact | `repro/q6k_x86.c` calls only llama.cpp functions |

## Mechanism: NOT identified
Tested and rejected:
1. `_mm256_maddubs_epi16` int16 saturation — worst pair sum 16002, fits
2. integral offset error in `q8sclsub` — implied discrepancy 0.0273, fractional
3. pre-multiplying scales into one fp32 `d` — modelling reproduces the CORRECT
   28799250.0

Unexplained observation: x86's low 11 mantissa bits are zero, and
`28794880 / 4096 = 7030` exactly vs a true `7031.066`.

## Build command for the reproducer
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build build --target ggml-cpu ggml-base -j

cc -O2 -I ggml/include -I ggml/src -I ggml/src/ggml-cpu \
   q6k_x86.c -o /tmp/q6k \
   build/ggml/src/libggml-cpu.a build/ggml/src/libggml-base.a \
   -lm -lstdc++ -fopenmp     # drop -fopenmp on macOS
/tmp/q6k
```

## Points the report should carry
- affected symbol and arch path
- the five input values and n=256
- the four numbers (exact / nearest / generic / x86) with bits
- the bound and that all 24 orderings were enumerated
- that the mechanism is unidentified — do not speculate
- that generic and NEON agree, so this is x86-AVX2-specific
