# x86 `ggml_vec_dot_q6_K_q8_K` loses ~1937x accuracy vs its own generic path

## Summary

On x86, the SIMD implementation of `ggml_vec_dot_q6_K_q8_K` returns a result
with relative error `1.5e-04` where `ggml_vec_dot_q6_K_q8_K_generic`, given
the identical quantized data, returns `7.8e-08`. The arm64 NEON path matches
the generic result exactly.

## Reproducer

`repro/q6k_x86.c` — uses only llama.cpp functions; no external harness.

```
x[87]  =  11429230.0f
x[140] = -774724.25f
x[143] =  510633.031f
y[86]  = -7282.96387f
y[143] =  65.4250183f
```
all other elements zero, `n = 256` (one q6_K super-block).

## Observed

| build | simd | generic | rel err (simd) | rel err (generic) |
|---|---|---|---|---|
| x86_64 baseline | 28794880 | 28799250 | 1.5166e-04 | 7.8283e-08 |
| x86_64 `-mavx2 -mfma` | 28794880 | 28799250 | 1.5166e-04 | 7.8283e-08 |
| x86_64 `-ffp-contract=off` | 28794880 | 28799250 | 1.5166e-04 | 7.8283e-08 |
| x86_64 `GGML_CPU_GENERIC=ON` | 28794880 | 28799250 | 1.5166e-04 | 7.8283e-08 |
| arm64 NEON | 28799250 | 28799250 | 7.8283e-08 | 7.8283e-08 |

Exact value (ggml's own `dequantize_row_q6_K`, summed in double):
`28799247.745513916`.

`simd - generic = -4370`.

## Ruled out

- **FMA contraction** — identical under `-ffp-contract=off`.
- **`bsums` inconsistency** — the x86 AVX2 kernel consumes `y[i].bsums` for
  its offset correction while the generic path recomputes from `qs`; they
  agree exactly (0/16 mismatches) for this input.
- **Harness involvement** — the reproducer calls only llama.cpp.
- **Non-portable input generation** — generation fingerprints are identical
  on both platforms, so the same seed denotes the same input.

## Not established

Whether `1.5e-04` is outside what q6_K's accumulation can legitimately
produce on inputs spanning many orders of magnitude. The generic path
achieving `7.8e-08` on the same data is suggestive but not proof: the two
accumulate differently by design. A hand-derived bound for this specific
input would settle it.

The input is adversarial (three weights spanning ~20x inside one super-block,
two activations), so this is a precision-loss question rather than an
obviously wrong answer.
