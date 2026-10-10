#ifndef CNN_STRIDE_HPP
#define CNN_STRIDE_HPP


#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.hpp" 
namespace cnn {

// Scoped inside cnn on purpose: it does NOT leak into the global namespace.
using namespace std;

/// Step sizes in the height and width directions. Stride(2) means 2 in both.
struct Stride {
    size_t h = 1;
    size_t w = 1;
    Stride() = default;
    explicit Stride(size_t both) : h(both), w(both) {}
    Stride(size_t sh, size_t sw) : h(sh), w(sw) {}
};

namespace stride_detail {

[[noreturn]] inline void fail(const char* who, const string& msg) {
    throw invalid_argument(string("stride::") + who + ": " + msg);
}

/// Non-throwing check of one dimension. Returns "" when valid, else the reason.
inline string checkWindow(size_t in, size_t kernel, size_t pad, size_t stride) {
    if (in == 0) return "input size must be positive";
    if (kernel == 0) return "kernel size must be positive";
    if (stride == 0) return "stride must be >= 1";
    if (pad > (numeric_limits<size_t>::max() - in) / 2) return "padding too large";
    if (in + 2 * pad < kernel) return "kernel is larger than the padded input";
    return string();
}

inline void requireValid(const Tensor& t, const char* who) {
    if (t.shape.size() != 4) fail(who, "tensor must be 4-D {N,C,H,W}");
    size_t total = 1;
    for (size_t d : t.shape) {
        if (d == 0) fail(who, "tensor has a zero dimension");
        if (total > numeric_limits<size_t>::max() / d) fail(who, "tensor too large");
        total *= d;
    }
    if (t.data.size() != total) fail(who, "data size does not match shape");
}

}  


/// True when (in, kernel, pad, stride) describes a valid sliding window.
/// If it is not valid and `reason` is non-null, the reason is written to it.
inline bool isValidWindow(size_t in, size_t kernel, size_t pad, size_t stride,
                          string* reason = nullptr) {
    const string why = stride_detail::checkWindow(in, kernel, pad, stride);
    if (!why.empty() && reason != nullptr) *reason = why;
    return why.empty();
}

/// Output length of one dimension: floor((in - kernel + 2*pad) / stride) + 1.
/// Requires in >= 1, kernel >= 1, stride >= 1 and kernel <= in + 2*pad.
inline size_t outputSize(size_t in, size_t kernel, size_t pad, size_t stride) {
    const string why = stride_detail::checkWindow(in, kernel, pad, stride);
    if (!why.empty()) stride_detail::fail("outputSize", why);
    return (in + 2 * pad - kernel) / stride + 1;
}

//Traverse the input tensor with a sliding window of size kernel_h x kernel_w, with padding pad and stride stride.

//Geometry of every window position for one image plane.
struct WindowGrid {
    size_t in_h = 0, in_w = 0;          
    size_t kernel_h = 0, kernel_w = 0;  
    size_t pad_h = 0, pad_w = 0;        
    Stride stride;
    size_t out_h = 0, out_w = 0;        

    //Top-left row/column of window (oh, ow) in PADDED coordinates.
    size_t paddedRow(size_t oh) const { return oh * stride.h; }
    size_t paddedCol(size_t ow) const { return ow * stride.w; }

    //Top-left row/column in INPUT coordinates (negative when inside padding).
    ptrdiff_t inputRow(size_t oh) const {
        return static_cast<ptrdiff_t>(oh * stride.h) - static_cast<ptrdiff_t>(pad_h);
    }
    ptrdiff_t inputCol(size_t ow) const {
        return static_cast<ptrdiff_t>(ow * stride.w) - static_cast<ptrdiff_t>(pad_w);
    }
};

//Build the grid with separate padding for height and width.
inline WindowGrid makeWindowGrid(size_t in_h, size_t in_w, size_t kernel_h, size_t kernel_w,
                                 size_t pad_h, size_t pad_w, Stride stride) {
    WindowGrid g;
    g.in_h = in_h;
    g.in_w = in_w;
    g.kernel_h = kernel_h;
    g.kernel_w = kernel_w;
    g.pad_h = pad_h;
    g.pad_w = pad_w;
    g.stride = stride;
    g.out_h = outputSize(in_h, kernel_h, pad_h, stride.h);
    g.out_w = outputSize(in_w, kernel_w, pad_w, stride.w);
    return g;
}

//Build the grid with the same padding P on all four sides.
inline WindowGrid makeWindowGrid(size_t in_h, size_t in_w, size_t kernel_h, size_t kernel_w,
                                 size_t pad, Stride stride) {
    return makeWindowGrid(in_h, in_w, kernel_h, kernel_w, pad, pad, stride);
}

// Visit every window in row-major order: f(oh, ow, padded_row, padded_col).
// padded_row / padded_col is the window's top-left corner in padded coordinates.
template <typename F>
void forEachWindow(const WindowGrid& g, F&& f) {
    for (size_t oh = 0; oh < g.out_h; ++oh) {
        for (size_t ow = 0; ow < g.out_w; ++ow) {
            f(oh, ow, g.paddedRow(oh), g.paddedCol(ow));
        }
    }
}

//Map a PADDED coordinate to an INPUT coordinate.
// Returns false (and leaves row/col untouched) when it lies in the padding,

inline bool paddedToInput(const WindowGrid& g, size_t padded_row, size_t padded_col, size_t& row,
                          size_t& col) {
    if (padded_row < g.pad_h || padded_col < g.pad_w) return false;
    const size_t r = padded_row - g.pad_h;
    const size_t c = padded_col - g.pad_w;
    if (r >= g.in_h || c >= g.in_w) return false;
    row = r;
    col = c;
    return true;
}

// Output shape of a windowed layer: {N, C, H_out, W_out}.
//A convolution replaces C with its number of filters afterwards.
inline vector<size_t> windowedOutputShape(const vector<size_t>& in_shape, size_t kernel_h,
                                          size_t kernel_w, size_t pad, Stride stride) {
    if (in_shape.size() != 4) stride_detail::fail("windowedOutputShape", "shape must be 4-D {N,C,H,W}");
    if (in_shape[0] == 0 || in_shape[1] == 0) {
        stride_detail::fail("windowedOutputShape", "N and C must be positive");
    }
    const WindowGrid g = makeWindowGrid(in_shape[2], in_shape[3], kernel_h, kernel_w, pad, stride);
    return vector<size_t>{in_shape[0], in_shape[1], g.out_h, g.out_w};
}



// Length after keeping every stride-th element starting at `offset`:
// Returns the number of elements that would be selected.
inline size_t subsampleSize(size_t in, size_t stride, size_t offset = 0) {
    if (in == 0) stride_detail::fail("subsampleSize", "input size must be positive");
    if (stride == 0) stride_detail::fail("subsampleSize", "stride must be >= 1");
    if (offset >= in) stride_detail::fail("subsampleSize", "offset must be smaller than the input size");
    return (in - offset - 1) / stride + 1;
}

inline Tensor subsample(const Tensor& in, Stride stride, size_t offset_h = 0, size_t offset_w = 0) {
    stride_detail::requireValid(in, "subsample");
    const size_t N = in.shape[0], C = in.shape[1], H = in.shape[2], W = in.shape[3];
    const size_t oh = subsampleSize(H, stride.h, offset_h);
    const size_t ow = subsampleSize(W, stride.w, offset_w);
    Tensor out(N, C, oh, ow);
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t i = 0; i < oh; ++i) {
                for (size_t j = 0; j < ow; ++j) {
                    const size_t src_row = offset_h + i * stride.h;
                    const size_t src_col = offset_w + j * stride.w;
                    out.data[((n * C + c) * oh + i) * ow + j] =
                        in.data[((n * C + c) * H + src_row) * W + src_col];
                }
            }
        }
    }
    return out;
}

}  // namespace cnn

#endif  // CNN_STRIDE_HPP