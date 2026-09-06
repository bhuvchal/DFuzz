#include "dfuzz/formats.h"
#include "dfuzz/reference.h"
#include <cmath>
#include <cstring>

namespace dfuzz {
namespace {

struct block_mxfp4 { uint8_t e;    uint8_t qs[QK_MXFP4/2]; };
struct block_nvfp4 { uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; uint8_t qs[QK_NVFP4/2]; };
struct block_q8_0  { uint16_t d;   int8_t  qs[QK8_0]; };

static_assert(sizeof(block_mxfp4) == 1 + QK_MXFP4/2, "mxfp4 layout");
static_assert(sizeof(block_nvfp4) == QK_NVFP4/QK_NVFP4_SUB + QK_NVFP4/2, "nvfp4 layout");
static_assert(sizeof(block_q8_0)  == sizeof(uint16_t) + QK8_0, "q8_0 layout");

double fp16_to_double(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    const uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t mant = (uint32_t)h & 0x3FFu;
    double mag;
    if (exp == 0)         mag = (double)mant * 5.9604644775390625e-08;
    else if (exp == 0x1F) mag = mant ? __builtin_nan("") : __builtin_inf();
    else                  mag = (1.0 + (double)mant / 1024.0) * __builtin_ldexp(1.0, (int)exp - 15);
    return sign ? -mag : mag;
}

} // namespace

// Mirrors ggml_e8m0_to_fp32_half: bits = (x-1) << 23, i.e. 2^(x-128), halved.
// Exact: a power of two in, a power of two out.
double e8m0_to_double_half(uint8_t e) {
    // x < 2 is a special case in ggml: fixed subnormal bit patterns
    // 0x00200000 = 2^-128, 0x00400000 = 2^-127. Reproduced exactly.
    if (e < 2) return __builtin_ldexp(1.0, -128 + (int)e);
    // Otherwise bits = (x-1) << 23, i.e. 2^(x-128).
    return __builtin_ldexp(1.0, (int)e - 128);
}

// Mirrors ggml_ue4m3_to_fp32 exactly, in fp64.
double ue4m3_to_double(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0;
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    const double raw = (exp == 0) ? __builtin_ldexp((double)man, -9)
                                  : __builtin_ldexp(1.0 + (double)man / 8.0, exp - 7);
    return raw * 0.5;
}

void dequantize_mxfp4(const void* vx, int nblocks, std::vector<double>& out) {
    const auto* x = static_cast<const block_mxfp4*>(vx);
    const int8_t* kv = kvalues_fp4();
    out.resize((size_t)nblocks * QK_MXFP4);
    for (int i = 0; i < nblocks; ++i) {
        const double d = e8m0_to_double_half(x[i].e);
        double* y = out.data() + (size_t)i * QK_MXFP4;
        // Low nibbles fill [0,16), high nibbles fill [16,32) -- matching the
        // kernel's sumi1/sumi2 split, not an interleave.
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            y[j]               = d * (double)kv[x[i].qs[j] & 0xf];
            y[j + QK_MXFP4/2]  = d * (double)kv[x[i].qs[j] >>  4];
        }
    }
}

void dequantize_nvfp4(const void* vx, int nblocks, std::vector<double>& out) {
    const auto* x = static_cast<const block_nvfp4*>(vx);
    const int8_t* kv = kvalues_fp4();
    out.resize((size_t)nblocks * QK_NVFP4);
    for (int i = 0; i < nblocks; ++i) {
        double* y = out.data() + (size_t)i * QK_NVFP4;
        for (int s = 0; s < 4; ++s) {
            const double d = ue4m3_to_double(x[i].d[s]);
            double* ys = y + (size_t)s * QK_NVFP4_SUB;
            for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                const uint8_t qv = x[i].qs[s*(QK_NVFP4_SUB/2) + j];
                ys[j]                    = d * (double)kv[qv & 0xf];
                ys[j + QK_NVFP4_SUB/2]   = d * (double)kv[qv >>  4];
            }
        }
    }
}

void dequantize_q8_0(const void* vy, int nblocks, std::vector<double>& out) {
    const auto* y = static_cast<const block_q8_0*>(vy);
    out.resize((size_t)nblocks * QK8_0);
    for (int i = 0; i < nblocks; ++i) {
        const double d = fp16_to_double(y[i].d);
        for (int j = 0; j < QK8_0; ++j)
            out[(size_t)i*QK8_0 + j] = d * (double)y[i].qs[j];
    }
}

int accumulation_depth(Format f, int n) {
    int d = 1;
    switch (f) {
        case Format::q4_K:  case Format::q6_K: d = n / QK_K;          break;
        case Format::mxfp4:                    d = n / QK_MXFP4;      break;
        case Format::nvfp4:                    d = n / QK_NVFP4_SUB;  break;
    }
    return d > 0 ? d : 1;
}

const char* format_name(Format f) {
    switch (f) {
        case Format::q4_K:  return "q4_K";
        case Format::q6_K:  return "q6_K";
        case Format::mxfp4: return "mxfp4";
        case Format::nvfp4: return "nvfp4";
    }
    return "?";
}

int block_bytes(Format f) {
    switch (f) {
        case Format::q4_K:  return 144;
        case Format::q6_K:  return 210;
        case Format::mxfp4: return (int)sizeof(block_mxfp4);
        case Format::nvfp4: return (int)sizeof(block_nvfp4);
    }
    return 0;
}

int block_elems(Format f) {
    switch (f) {
        case Format::q4_K:  case Format::q6_K: return QK_K;
        case Format::mxfp4: return QK_MXFP4;
        case Format::nvfp4: return QK_NVFP4;
    }
    return 0;
}

} // namespace dfuzz
