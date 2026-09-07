#include "dfuzz/oracle.h"
#include "dfuzz/reference.h"
#include "dfuzz/init.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

extern "C" {
void quantize_row_mxfp4_ref(const float*, void*, int64_t);
void quantize_row_nvfp4_ref(const float*, void*, int64_t);
void quantize_row_q4_K_ref (const float*, void*, int64_t);
void quantize_row_q6_K_ref (const float*, void*, int64_t);
void quantize_row_q8_0_ref (const float*, void*, int64_t);
void quantize_row_q8_K_ref (const float*, void*, int64_t);
void ggml_vec_dot_mxfp4_q8_0        (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_mxfp4_q8_0_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_nvfp4_q8_0        (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_nvfp4_q8_0_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_q4_K_q8_K         (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_q4_K_q8_K_generic (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_q6_K_q8_K         (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_q6_K_q8_K_generic (int,float*,size_t,const void*,size_t,const void*,size_t,int);
}

namespace dfuzz {

const char* axis_name(Axis a) {
    switch (a) {
        case Axis::SimdVsRef:     return "simd-vs-ref";
        case Axis::GenericVsRef:  return "generic-vs-ref";
        case Axis::SimdVsGeneric: return "simd-vs-generic";
    }
    return "?";
}

Signature make_signature(Format fmt, Axis axis, double ratio, bool nonfinite) {
    Signature s;
    s.fmt  = fmt;
    s.axis = axis;
    s.nonfinite = nonfinite;
    // log10 bucket of the violation. Shrinking moves the ratio; the decade is
    // stable enough to identify the bug and strict enough to catch a swap.
    s.severity_decade = (ratio > 0.0 && std::isfinite(ratio))
                      ? (int)std::floor(std::log10(ratio)) : 99;
    return s;
}

struct Oracle::Scratch {
    // Grow-only buffers, reused across every evaluation.
    std::vector<uint8_t> xq, yq;
    std::vector<double>  xd, yd;
};

Oracle::Oracle() : s_(new Scratch) {}
Oracle::~Oracle() { delete s_; }
Oracle::Oracle(Oracle&& o) noexcept
    : s_(std::exchange(o.s_, nullptr)),
      evals_(std::exchange(o.evals_, 0)),
      allocs_(std::exchange(o.allocs_, 0)) {}
Oracle& Oracle::operator=(Oracle&& o) noexcept {
    if (this != &o) {
        delete s_;
        s_      = std::exchange(o.s_, nullptr);
        evals_  = std::exchange(o.evals_, 0);
        allocs_ = std::exchange(o.allocs_, 0);
    }
    return *this;
}

namespace {

// The activation side uses q8_0 for the fp4 formats and q8_K for the K-quants.
bool uses_q8_K(Format f) { return f == Format::q4_K || f == Format::q6_K; }
int  y_block_bytes(Format f) { return uses_q8_K(f) ? 292 : 34; }
int  y_block_elems(Format f) { return uses_q8_K(f) ? 256 : 32; }

} // namespace

Outcome Oracle::evaluate(const Case& c) {
    require_initialized("Oracle::evaluate");
    ++evals_;

    Outcome out;
    const int n = c.n();
    const int nbx = n / block_elems(c.fmt);
    const int nby = n / y_block_elems(c.fmt);
    if (n == 0 || nbx == 0 || nby == 0) return out;

    // Grow-only: capacity is retained across calls, so steady state allocates
    // nothing. Counted so the benchmark can prove it.
    const size_t need_x = (size_t)nbx * block_bytes(c.fmt);
    const size_t need_y = (size_t)nby * y_block_bytes(c.fmt);
    if (s_->xq.size() < need_x) { s_->xq.resize(need_x); ++allocs_; }
    if (s_->yq.size() < need_y) { s_->yq.resize(need_y); ++allocs_; }

    switch (c.fmt) {
        case Format::mxfp4: quantize_row_mxfp4_ref(c.x.data(), s_->xq.data(), n); break;
        case Format::nvfp4: quantize_row_nvfp4_ref(c.x.data(), s_->xq.data(), n); break;
        case Format::q4_K:  quantize_row_q4_K_ref (c.x.data(), s_->xq.data(), n); break;
        case Format::q6_K:  quantize_row_q6_K_ref (c.x.data(), s_->xq.data(), n); break;
    }
    if (uses_q8_K(c.fmt)) quantize_row_q8_K_ref(c.y.data(), s_->yq.data(), n);
    else                  quantize_row_q8_0_ref(c.y.data(), s_->yq.data(), n);

    float simd = 0, gen = 0;
    switch (c.fmt) {
        case Format::mxfp4:
            ggml_vec_dot_mxfp4_q8_0        (n,&simd,0,s_->xq.data(),0,s_->yq.data(),0,1);
            ggml_vec_dot_mxfp4_q8_0_generic(n,&gen ,0,s_->xq.data(),0,s_->yq.data(),0,1); break;
        case Format::nvfp4:
            ggml_vec_dot_nvfp4_q8_0        (n,&simd,0,s_->xq.data(),0,s_->yq.data(),0,1);
            ggml_vec_dot_nvfp4_q8_0_generic(n,&gen ,0,s_->xq.data(),0,s_->yq.data(),0,1); break;
        case Format::q4_K:
            ggml_vec_dot_q4_K_q8_K         (n,&simd,0,s_->xq.data(),0,s_->yq.data(),0,1);
            ggml_vec_dot_q4_K_q8_K_generic (n,&gen ,0,s_->xq.data(),0,s_->yq.data(),0,1); break;
        case Format::q6_K:
            ggml_vec_dot_q6_K_q8_K         (n,&simd,0,s_->xq.data(),0,s_->yq.data(),0,1);
            ggml_vec_dot_q6_K_q8_K_generic (n,&gen ,0,s_->xq.data(),0,s_->yq.data(),0,1); break;
    }

    switch (c.fmt) {
        case Format::mxfp4: dequantize_mxfp4(s_->xq.data(), nbx, s_->xd); break;
        case Format::nvfp4: dequantize_nvfp4(s_->xq.data(), nbx, s_->xd); break;
        case Format::q4_K:  dequantize_q4_K (s_->xq.data(), nbx, s_->xd); break;
        case Format::q6_K:  dequantize_q6_K (s_->xq.data(), nbx, s_->xd); break;
    }
    if (uses_q8_K(c.fmt)) dequantize_q8_K(s_->yq.data(), nby, s_->yd);
    else                  dequantize_q8_0(s_->yq.data(), nby, s_->yd);

    // Sub-block size at which the kernel forms fp32 partial sums. Passing it
    // lets the reference record the peak intermediate magnitude, which is what
    // the kernel's roundings actually act on -- for q4_K that is far larger
    // than sum|x_i y_i| because a big offset term is subtracted from a big
    // scaled sum.
    const int sub = (c.fmt == Format::q4_K || c.fmt == Format::q6_K) ? 32
                  : (c.fmt == Format::nvfp4 ? QK_NVFP4_SUB : QK_MXFP4);
    RefResult ref = reference_dot_blocked(s_->xd, s_->yd, sub);
    const int depth = accumulation_depth(c.fmt, n);

    // q4_K reconstructs x = d*scale*q - dmin*min. The kernel therefore forms
    // a large positive scaled sum and subtracts a large offset term, and
    // rounds BOTH in fp32. Neither appears in sum|x_i y_i|, which only sees
    // the differences. Measured on failing cases the offset term ran 30-60x
    // sum|x_i y_i|, which is precisely the gap that made the bound too tight
    // for mixed-magnitude inputs.
    //
    // Only q4_K carries an offset; q6_K, mxfp4 and nvfp4 are pure scale.
    if (c.fmt == Format::q4_K) {
        ref.intermediate_mag = std::max(ref.intermediate_mag,
                                        q4k_offset_magnitude(s_->xq.data(), s_->yq.data(), nbx));
    }

    out.simd = simd; out.generic = gen; out.reference = ref.dot; out.abs_sum = ref.abs_sum;

    struct { Axis ax; double a, b; } axes[] = {
        {Axis::SimdVsRef,     (double)simd, ref.dot},
        {Axis::GenericVsRef,  (double)gen,  ref.dot},
        {Axis::SimdVsGeneric, (double)simd, (double)gen},
    };
    for (auto& e : axes) {
        // depth*256 with qk=256 reproduces the per-format depth inside check().
        const Verdict v = check(e.a, e.b, ref.intermediate_mag, depth * 256, 256);
        if (v.ratio > out.ratio || !out.failed) {
            if (v.ratio > out.ratio) { out.ratio = v.ratio; out.worst = e.ax; }
        }
        if (!v.ok) out.failed = true;
    }
    const bool nonfinite = !std::isfinite(simd) || !std::isfinite(gen) || !std::isfinite(ref.dot);
    out.sig = make_signature(c.fmt, out.worst, out.ratio, nonfinite);
    return out;
}

bool Oracle::still_fails(const Case& c, const Signature& want) {
    const Outcome o = evaluate(c);
    return o.failed && o.sig == want;
}

} // namespace dfuzz
