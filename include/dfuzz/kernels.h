#pragma once
#include <cstddef>
#include <cstdint>

// Direct declarations of llama.cpp internal CPU kernels.
// We deliberately do NOT include ggml's quants.h: it pulls in arch-fallback.h,
// which #defines the _generic names onto the plain names under some configs.
// Declaring the ABI ourselves keeps the two variants distinguishable everywhere.
extern "C" {

// Arch-dispatched (NEON on arm64, AVX2/512 on x86) and portable scalar.
void ggml_vec_dot_q4_K_q8_K        (int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K_generic(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K        (int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K_generic(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);

// Reference-side quantizers (ggml-quants.c, always scalar).
void quantize_row_q4_K_ref(const float* x, void* y, int64_t k);
void quantize_row_q6_K_ref(const float* x, void* y, int64_t k);
void quantize_row_q8_K_ref(const float* x, void* y, int64_t k);

} // extern "C"
