// Standalone reproducer: x86 ggml_vec_dot_q6_K_q8_K diverges from
// ggml_vec_dot_q6_K_q8_K_generic on the same quantized data.
//
// Uses only llama.cpp's own functions -- no part of the DFuzz harness.
// Build (from the repo root, after cmake --build build):
//   cc -O2 -I third_party/llama.cpp/ggml/include \
//      -I third_party/llama.cpp/ggml/src \
//      -I third_party/llama.cpp/ggml/src/ggml-cpu \
//      repro/q6k_x86.c -o /tmp/q6k \
//      build/third_party/llama.cpp/ggml/src/libggml-cpu.a \
//      build/third_party/llama.cpp/ggml/src/libggml-base.a -lm -lstdc++ -fopenmp
// (drop -fopenmp on macOS, where ggml does not link it)
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

void ggml_cpu_init(void);
void quantize_row_q6_K_ref(const float*, void*, int64_t);
void quantize_row_q8_K_ref(const float*, void*, int64_t);
void ggml_vec_dot_q6_K_q8_K        (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_q6_K_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
void dequantize_row_q6_K(const void*, float*, int64_t);

#define N 256

int main(void) {
    ggml_cpu_init();

    float x[N] = {0}, y[N] = {0};
    // Minimized by the DFuzz shrinker from q6_K/mixed-magnitude seed
    // 0x9e3579b87e4a7c19: 3 nonzero weights, 2 nonzero activations.
    x[87]  =  11429230.0f;
    x[140] = -774724.25f;
    x[143] =  510633.031f;
    y[86]  = -7282.96387f;
    y[143] =  65.4250183f;

    static unsigned char xq[210], yq[292];
    quantize_row_q6_K_ref(x, xq, N);
    quantize_row_q8_K_ref(y, yq, N);

    float simd = 0.0f, gen = 0.0f;
    ggml_vec_dot_q6_K_q8_K        (N, &simd, 0, xq, 0, yq, 0, 1);
    ggml_vec_dot_q6_K_q8_K_generic(N, &gen,  0, xq, 0, yq, 0, 1);

    // Exact expected value: dequantize with ggml's own dequantizer, then sum
    // in double. This is the value both kernels are approximating.
    static float xd[N];
    dequantize_row_q6_K(xq, xd, N);
    float yd_scale;
    memcpy(&yd_scale, yq, sizeof(float));
    const signed char* yqs = (const signed char*)(yq + 4);
    double exact = 0.0;
    for (int i = 0; i < N; ++i)
        exact += (double)xd[i] * ((double)yd_scale * (double)yqs[i]);

    printf("simd     = %.9g   rel err vs exact = %.4e\n",
           (double)simd, fabs((double)simd - exact) / fabs(exact));
    printf("generic  = %.9g   rel err vs exact = %.4e\n",
           (double)gen,  fabs((double)gen  - exact) / fabs(exact));
    printf("exact    = %.17g\n", exact);
    printf("simd - generic = %.9g\n", (double)simd - (double)gen);

    const double rs = fabs((double)simd - exact) / fabs(exact);
    const double rg = fabs((double)gen  - exact) / fabs(exact);
    printf("\n%s: simd is %.0fx less accurate than generic\n",
           rs > 100.0 * rg ? "DIVERGENCE" : "consistent",
           rg > 0 ? rs / rg : 0.0);
    return 0;
}
