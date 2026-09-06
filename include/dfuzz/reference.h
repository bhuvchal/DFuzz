#pragma once
#include <cstdint>
#include <vector>

namespace dfuzz {

constexpr int QK_K = 256;

// --- Exact dequantization -------------------------------------------------
// Mirrors ggml's bit layout exactly. These produce the **real numbers the
// quantized format denotes**, not an approximation of the original floats.
// Quantization loss is not what we measure; accumulation order is. Both the
// reference and the kernel must consume identical dequantized values.

// Unpacks the 6-bit scale/min pairs packed into 12 bytes.
// Mirrors get_scale_min_k4() in ggml-quants.c.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

// x = d * scale * q - dmin * min, per 32-element sub-block.
void dequantize_q4_K(const void* vx, int nblocks, std::vector<double>& out);
// x = d * scale8 * q, per 16-element sub-block; quants are signed 6-bit.
void dequantize_q6_K(const void* vx, int nblocks, std::vector<double>& out);
// y = d * q, one fp32 scale per super-block.
void dequantize_q8_K(const void* vy, int nblocks, std::vector<double>& out);

// --- fp64 reference -------------------------------------------------------
struct RefResult {
    double dot;      // exact-as-practical dot product in fp64
    double abs_sum;  // sum |x_i * y_i| -- drives the error bound
};

// Straight serial fp64 accumulation over exactly-dequantized values.
RefResult reference_dot(const std::vector<double>& x, const std::vector<double>& y);

} // namespace dfuzz
