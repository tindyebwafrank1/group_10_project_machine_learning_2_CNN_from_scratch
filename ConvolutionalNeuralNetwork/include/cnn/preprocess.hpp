#ifndef CNN_PREPROCESS_HPP
#define CNN_PREPROCESS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.hpp"  // Agatha: Tensor{ vector<double> data; vector<size_t> shape{N,C,H,W}; Tensor(n,c,h,w) }

namespace cnn {

enum class Interpolation { Nearest, Bilinear };
enum class Normalization { None, Scale, Standardize, MinMax };

//One-call preprocessing recipe, applied in this order:
//reverse_channels -> target_channels -> resize -> normalization.
struct PreprocessConfig {
    bool reverse_channels = false;          ///< RGB <-> BGR
    std::size_t target_channels = 0;        ///< 0 = keep, 1 = grayscale, 3 = RGB
    std::size_t target_height = 0;          ///< 0 = keep (set both or neither)
    std::size_t target_width = 0;
    Interpolation interpolation = Interpolation::Bilinear;
    Normalization normalization = Normalization::Scale;
    double max_value = 255.0;               ///< Scale/Standardize: x / max_value
    std::vector<double> mean;               ///< Standardize: one per output channel
    std::vector<double> stddev;             ///< Standardize: one per output channel, > 0
    double range_min = 0.0;                 ///< MinMax target range
    double range_max = 1.0;
};

struct ChannelStats {
    std::vector<double> mean;
    std::vector<double> stddev;  ///< population std; 1.0 where a channel is constant
};

namespace detail {

[[noreturn]] inline void fail(const char* who, const std::string& msg) {
    throw std::invalid_argument(std::string("preprocess::") + who + ": " + msg);
}

inline std::size_t checkedProduct(std::size_t a, std::size_t b, const char* who) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) fail(who, "tensor too large");
    return a * b;
}

/// Index(n,c,h,w) = n*(C*H*W) + c*(H*W) + h*W + w
inline std::size_t flat(const Tensor& t, std::size_t n, std::size_t c, std::size_t h,
                        std::size_t w) {
    return ((n * t.shape[1] + c) * t.shape[2] + h) * t.shape[3] + w;
}

inline void requireValid(const Tensor& t, const char* who) {
    if (t.shape.size() != 4) fail(who, "tensor must be 4-D {N,C,H,W}");
    std::size_t total = 1;
    for (std::size_t d : t.shape) {
        if (d == 0) fail(who, "tensor has a zero dimension");
        total = checkedProduct(total, d, who);
    }
    if (t.data.size() != total) fail(who, "data size does not match shape");
}

inline void requireFinite(const Tensor& t, const char* who) {
    for (double v : t.data) {
        if (!std::isfinite(v)) fail(who, "tensor contains NaN or infinity");
    }
}

inline void check(const Tensor& t, const char* who) {
    requireValid(t, who);
    requireFinite(t, who);
}

inline Tensor makeOutput(std::size_t n, std::size_t c, std::size_t h, std::size_t w,
                         const char* who) {
    static_cast<void>(checkedProduct(checkedProduct(checkedProduct(n, c, who), h, who), w, who));
    return Tensor(n, c, h, w);
}

}  // namespace detail

// layout / channel order

template <typename T>
Tensor fromInterleaved(const T* pixels, std::size_t count, std::size_t height, std::size_t width,
                       std::size_t channels) {
    if (pixels == nullptr || count == 0) detail::fail("fromInterleaved", "empty buffer");
    if (height == 0 || width == 0) detail::fail("fromInterleaved", "zero dimension");
    if (channels != 1 && channels != 3) detail::fail("fromInterleaved", "channels must be 1 or 3");
    if (width > std::numeric_limits<std::size_t>::max() / height / channels) {
        detail::fail("fromInterleaved", "image too large");
    }
    if (count != height * width * channels) detail::fail("fromInterleaved", "size mismatch");
    Tensor out(1, channels, height, width);
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            for (std::size_t c = 0; c < channels; ++c) {
                out.data[(c * height + y) * width + x] =
                    static_cast<double>(pixels[(y * width + x) * channels + c]);
            }
        }
    }
    return out;
}

/// Sample n of a tensor -> interleaved HWC vector (for saving/inspecting images).
inline std::vector<double> toInterleaved(const Tensor& in, std::size_t n = 0) {
    detail::requireValid(in, "toInterleaved");
    if (n >= in.shape[0]) detail::fail("toInterleaved", "sample index out of range");
    const std::size_t C = in.shape[1], H = in.shape[2], W = in.shape[3];
    std::vector<double> out(H * W * C);
    for (std::size_t y = 0; y < H; ++y)
        for (std::size_t x = 0; x < W; ++x)
            for (std::size_t c = 0; c < C; ++c)
                out[(y * W + x) * C + c] = in.data[detail::flat(in, n, c, y, x)];
    return out;
}

/// out channel i = in channel order[i]. `order` must be a permutation of 0..C-1.
inline Tensor permuteChannels(const Tensor& in, const std::vector<std::size_t>& order) {
    detail::check(in, "permuteChannels");
    const std::size_t N = in.shape[0], C = in.shape[1], H = in.shape[2], W = in.shape[3];
    if (order.size() != C) detail::fail("permuteChannels", "order size must equal channel count");
    std::vector<bool> seen(C, false);
    for (std::size_t src : order) {
        if (src >= C || seen[src]) detail::fail("permuteChannels", "order is not a permutation");
        seen[src] = true;
    }
    Tensor out(N, C, H, W);
    const std::size_t plane = H * W;
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t c = 0; c < C; ++c) {
            const double* src = in.data.data() + detail::flat(in, n, order[c], 0, 0);
            std::copy(src, src + plane, out.data.data() + detail::flat(out, n, c, 0, 0));
        }
    }
    return out;
}

/// RGB <-> BGR (reverses channel order).
inline Tensor reverseChannels(const Tensor& in) {
    detail::requireValid(in, "reverseChannels");
    std::vector<std::size_t> order(in.shape[1]);
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = order.size() - 1 - i;
    return permuteChannels(in, order);
}

// colour convert

/// {N,3,H,W} RGB -> {N,1,H,W} using luminance 0.299 R + 0.587 G + 0.114 B.
inline Tensor rgbToGrayscale(const Tensor& in) {
    detail::check(in, "rgbToGrayscale");
    if (in.shape[1] != 3) detail::fail("rgbToGrayscale", "input must have 3 channels");
    const std::size_t N = in.shape[0], H = in.shape[2], W = in.shape[3];
    const std::size_t plane = H * W;
    Tensor out(N, 1, H, W);
    for (std::size_t n = 0; n < N; ++n) {
        const std::size_t base = detail::flat(in, n, 0, 0, 0);
        for (std::size_t i = 0; i < plane; ++i) {
            out.data[n * plane + i] = 0.299 * in.data[base + i] +
                                      0.587 * in.data[base + plane + i] +
                                      0.114 * in.data[base + 2 * plane + i];
        }
    }
    return out;
}

/// {N,1,H,W} -> {N,3,H,W} by replicating the single channel.
inline Tensor grayscaleToRgb(const Tensor& in) {
    detail::check(in, "grayscaleToRgb");
    if (in.shape[1] != 1) detail::fail("grayscaleToRgb", "input must have 1 channel");
    const std::size_t N = in.shape[0], H = in.shape[2], W = in.shape[3];
    const std::size_t plane = H * W;
    Tensor out(N, 3, H, W);
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t c = 0; c < 3; ++c) {
            std::copy(in.data.begin() + static_cast<std::ptrdiff_t>(n * plane),
                      in.data.begin() + static_cast<std::ptrdiff_t>((n + 1) * plane),
                      out.data.begin() + static_cast<std::ptrdiff_t>((n * 3 + c) * plane));
        }
    }
    return out;
}

/// Convert to 1 or 3 channels; returns a copy unchanged if already there.
inline Tensor toChannels(const Tensor& in, std::size_t target) {
    detail::requireValid(in, "toChannels");
    if (target != 1 && target != 3) detail::fail("toChannels", "target must be 1 or 3");
    const std::size_t C = in.shape[1];
    if (C == target) return in;
    if (C == 3 && target == 1) return rgbToGrayscale(in);
    if (C == 1 && target == 3) return grayscaleToRgb(in);
    detail::fail("toChannels", "input must have 1 or 3 channels");
}

// resizing

/// Resize H and W of every sample/channel.
/// Nearest: pixel centre mapping. Bilinear: half-pixel centres, edges clamped.
inline Tensor resize(const Tensor& in, std::size_t outH, std::size_t outW,
                     Interpolation mode = Interpolation::Bilinear) {
    detail::check(in, "resize");
    if (outH == 0 || outW == 0) detail::fail("resize", "target size must be positive");
    const std::size_t N = in.shape[0], C = in.shape[1], H = in.shape[2], W = in.shape[3];
    Tensor out = detail::makeOutput(N, C, outH, outW, "resize");
    const double scaleY = static_cast<double>(H) / static_cast<double>(outH);
    const double scaleX = static_cast<double>(W) / static_cast<double>(outW);

    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t c = 0; c < C; ++c) {
            const double* src = in.data.data() + detail::flat(in, n, c, 0, 0);
            double* dst = out.data.data() + detail::flat(out, n, c, 0, 0);
            for (std::size_t y = 0; y < outH; ++y) {
                if (mode == Interpolation::Nearest) {
                    const std::size_t sy = std::min(
                        H - 1, static_cast<std::size_t>(
                                   std::floor((static_cast<double>(y) + 0.5) * scaleY)));
                    for (std::size_t x = 0; x < outW; ++x) {
                        const std::size_t sx = std::min(
                            W - 1, static_cast<std::size_t>(
                                       std::floor((static_cast<double>(x) + 0.5) * scaleX)));
                        dst[y * outW + x] = src[sy * W + sx];
                    }
                } else {
                    double fy = (static_cast<double>(y) + 0.5) * scaleY - 0.5;
                    fy = std::min(std::max(fy, 0.0), static_cast<double>(H - 1));
                    const std::size_t y0 = static_cast<std::size_t>(fy);
                    const std::size_t y1 = std::min(y0 + 1, H - 1);
                    const double wy = fy - static_cast<double>(y0);
                    for (std::size_t x = 0; x < outW; ++x) {
                        double fx = (static_cast<double>(x) + 0.5) * scaleX - 0.5;
                        fx = std::min(std::max(fx, 0.0), static_cast<double>(W - 1));
                        const std::size_t x0 = static_cast<std::size_t>(fx);
                        const std::size_t x1 = std::min(x0 + 1, W - 1);
                        const double wx = fx - static_cast<double>(x0);
                        const double top = src[y0 * W + x0] * (1.0 - wx) + src[y0 * W + x1] * wx;
                        const double bot = src[y1 * W + x0] * (1.0 - wx) + src[y1 * W + x1] * wx;
                        dst[y * outW + x] = top * (1.0 - wy) + bot * wy;
                    }
                }
            }
        }
    }
    return out;
}

// normalization

/// x / maxValue (maxValue > 0). Use 255 for 8-bit images.
inline Tensor scaleToUnit(const Tensor& in, double maxValue = 255.0) {
    detail::check(in, "scaleToUnit");
    if (!(maxValue > 0.0) || !std::isfinite(maxValue)) {
        detail::fail("scaleToUnit", "maxValue must be finite and > 0");
    }
    Tensor out = in;
    for (double& v : out.data) v /= maxValue;
    return out;
}

/// (x - mean[c]) / stddev[c], one mean/stddev per channel; every stddev > 0.
inline Tensor standardize(const Tensor& in, const std::vector<double>& mean,
                          const std::vector<double>& stddev) {
    detail::check(in, "standardize");
    const std::size_t N = in.shape[0], C = in.shape[1], plane = in.shape[2] * in.shape[3];
    if (mean.size() != C || stddev.size() != C) {
        detail::fail("standardize", "mean/stddev size must equal channel count");
    }
    for (std::size_t c = 0; c < C; ++c) {
        if (!std::isfinite(mean[c])) detail::fail("standardize", "mean must be finite");
        if (!(stddev[c] > 0.0) || !std::isfinite(stddev[c])) {
            detail::fail("standardize", "stddev must be finite and > 0");
        }
    }
    Tensor out = in;
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t c = 0; c < C; ++c) {
            double* p = out.data.data() + detail::flat(out, n, c, 0, 0);
            for (std::size_t i = 0; i < plane; ++i) p[i] = (p[i] - mean[c]) / stddev[c];
        }
    }
    return out;
}

/// Per-sample min-max scaling into [lo, hi]. A constant sample maps to lo.
inline Tensor minMaxNormalize(const Tensor& in, double lo = 0.0, double hi = 1.0) {
    detail::check(in, "minMaxNormalize");
    if (!(lo < hi) || !std::isfinite(lo) || !std::isfinite(hi)) {
        detail::fail("minMaxNormalize", "need finite lo < hi");
    }
    const std::size_t N = in.shape[0];
    const std::size_t per = in.shape[1] * in.shape[2] * in.shape[3];
    Tensor out = in;
    for (std::size_t n = 0; n < N; ++n) {
        double* p = out.data.data() + n * per;
        const double mn = *std::min_element(p, p + per);
        const double mx = *std::max_element(p, p + per);
        for (std::size_t i = 0; i < per; ++i) {
            p[i] = (mx == mn) ? lo : lo + (p[i] - mn) * (hi - lo) / (mx - mn);
        }
    }
    return out;
}

/// Per-channel mean and population std over N,H,W (std = 1.0 for constant channels,
/// so the result can always be passed straight to standardize()).
inline ChannelStats computeChannelStats(const Tensor& in) {
    detail::check(in, "computeChannelStats");
    const std::size_t N = in.shape[0], C = in.shape[1], plane = in.shape[2] * in.shape[3];
    const double count = static_cast<double>(N * plane);
    ChannelStats s;
    s.mean.assign(C, 0.0);
    s.stddev.assign(C, 1.0);
    for (std::size_t c = 0; c < C; ++c) {
        double sum = 0.0;
        for (std::size_t n = 0; n < N; ++n) {
            const double* p = in.data.data() + detail::flat(in, n, c, 0, 0);
            for (std::size_t i = 0; i < plane; ++i) sum += p[i];
        }
        const double mean = sum / count;
        double sq = 0.0;
        for (std::size_t n = 0; n < N; ++n) {
            const double* p = in.data.data() + detail::flat(in, n, c, 0, 0);
            for (std::size_t i = 0; i < plane; ++i) sq += (p[i] - mean) * (p[i] - mean);
        }
        const double sd = std::sqrt(sq / count);
        s.mean[c] = mean;
        s.stddev[c] = (sd > 1e-12) ? sd : 1.0;
    }
    return s;
}

// batches

/// Concatenate tensors along N. All must share C, H, W.
inline Tensor stackBatch(const std::vector<Tensor>& items) {
    if (items.empty()) detail::fail("stackBatch", "empty list");
    std::size_t totalN = 0;
    for (const Tensor& t : items) {
        detail::requireValid(t, "stackBatch");
        if (t.shape[1] != items[0].shape[1] || t.shape[2] != items[0].shape[2] ||
            t.shape[3] != items[0].shape[3]) {
            detail::fail("stackBatch", "all tensors must share C, H, W");
        }
        totalN += t.shape[0];
    }
    Tensor out = detail::makeOutput(totalN, items[0].shape[1], items[0].shape[2],
                                    items[0].shape[3], "stackBatch");
    std::size_t offset = 0;
    for (const Tensor& t : items) {
        std::copy(t.data.begin(), t.data.end(),
                  out.data.begin() + static_cast<std::ptrdiff_t>(offset));
        offset += t.data.size();
    }
    return out;
}

//pipeline

inline void validateConfig(const PreprocessConfig& cfg) {
    if ((cfg.target_height == 0) != (cfg.target_width == 0)) {
        detail::fail("validateConfig", "set both target_height and target_width, or neither");
    }
    if (cfg.target_channels != 0 && cfg.target_channels != 1 && cfg.target_channels != 3) {
        detail::fail("validateConfig", "target_channels must be 0, 1 or 3");
    }
    if ((cfg.normalization == Normalization::Scale ||
         cfg.normalization == Normalization::Standardize) &&
        (!(cfg.max_value > 0.0) || !std::isfinite(cfg.max_value))) {
        detail::fail("validateConfig", "max_value must be finite and > 0");
    }
    if (cfg.normalization == Normalization::MinMax && !(cfg.range_min < cfg.range_max)) {
        detail::fail("validateConfig", "need range_min < range_max");
    }
}

/// Full recipe. Input and output are {N,C,H,W}; the input is not modified.
/// Standardize = (x / max_value - mean) / stddev.
inline Tensor preprocess(const Tensor& input, const PreprocessConfig& cfg) {
    validateConfig(cfg);
    detail::check(input, "preprocess");
    Tensor cur = input;
    if (cfg.reverse_channels) cur = reverseChannels(cur);
    if (cfg.target_channels != 0) cur = toChannels(cur, cfg.target_channels);
    if (cfg.target_height != 0) {
        cur = resize(cur, cfg.target_height, cfg.target_width, cfg.interpolation);
    }
    switch (cfg.normalization) {
        case Normalization::None:
            break;
        case Normalization::Scale:
            cur = scaleToUnit(cur, cfg.max_value);
            break;
        case Normalization::Standardize:
            cur = standardize(scaleToUnit(cur, cfg.max_value), cfg.mean, cfg.stddev);
            break;
        case Normalization::MinMax:
            cur = minMaxNormalize(cur, cfg.range_min, cfg.range_max);
            break;
    }
    return cur;
}

}  // namespace cnn

#endif  // CNN_PREPROCESS_HPP