
#ifndef CNN_PREPROCESS_HPP
#define CNN_PREPROCESS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.hpp"

using namespace std;

namespace cnn {

enum class Interpolation {
    Nearest,
    Bilinear
};

enum class Normalization {
    None,
    Scale,
    Standardize,
    MinMax
};

struct PreprocessConfig {
    bool reverse_channels = false;
    size_t target_channels = 0;
    size_t target_height = 0;
    size_t target_width = 0;

    Interpolation interpolation = Interpolation::Bilinear;
    Normalization normalization = Normalization::Scale;

    double max_value = 255.0;

    vector<double> mean;
    vector<double> stddev;

    double range_min = 0.0;
    double range_max = 1.0;
};

struct ChannelStats {
    vector<double> mean;
    vector<double> stddev;
};

namespace detail {

[[noreturn]] inline void fail(const char* who, const string& msg) {
    throw invalid_argument(string("preprocess::") + who + ": " + msg);
}

inline size_t checkedProduct(size_t a, size_t b, const char* who) {
    if (a != 0 && b > numeric_limits<size_t>::max() / a) {
        fail(who, "tensor too large");
    }

    return a * b;
}

inline size_t flat(
    const Tensor& t,
    size_t n,
    size_t c,
    size_t h,
    size_t w
) {
    return ((n * t.shape[1] + c) * t.shape[2] + h)
           * t.shape[3] + w;
}

inline void requireValid(const Tensor& t, const char* who) {
    if (t.shape.size() != 4) {
        fail(who, "tensor must be 4-D {N,C,H,W}");
    }

    size_t total = 1;

    for (size_t d : t.shape) {
        if (d == 0) {
            fail(who, "tensor has a zero dimension");
        }

        total = checkedProduct(total, d, who);
    }

    if (t.data.size() != total) {
        fail(who, "data size does not match shape");
    }
}

inline void requireFinite(const Tensor& t, const char* who) {
    for (double v : t.data) {
        if (!isfinite(v)) {
            fail(who, "tensor contains NaN or infinity");
        }
    }
}

inline void check(const Tensor& t, const char* who) {
    requireValid(t, who);
    requireFinite(t, who);
}

inline Tensor makeOutput(
    size_t n,
    size_t c,
    size_t h,
    size_t w,
    const char* who
) {
    size_t total = checkedProduct(n, c, who);
    total = checkedProduct(total, h, who);
    static_cast<void>(checkedProduct(total, w, who));

    return Tensor(n, c, h, w);
}

} // namespace detail

// Convert an interleaved HWC pixel buffer to a {1,C,H,W} tensor.
// Values are copied unchanged.
template <typename T>
Tensor fromInterleaved(
    const T* pixels,
    size_t count,
    size_t height,
    size_t width,
    size_t channels
) {
    if (pixels == nullptr || count == 0) {
        detail::fail("fromInterleaved", "empty buffer");
    }

    if (height == 0 || width == 0) {
        detail::fail("fromInterleaved", "zero dimension");
    }

    if (channels != 1 && channels != 3) {
        detail::fail("fromInterleaved", "channels must be 1 or 3");
    }

    if (width > numeric_limits<size_t>::max() / height / channels) {
        detail::fail("fromInterleaved", "image too large");
    }

    if (count != height * width * channels) {
        detail::fail("fromInterleaved", "size mismatch");
    }

    Tensor out(1, channels, height, width);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            for (size_t c = 0; c < channels; ++c) {
                out.data[(c * height + y) * width + x] =
                    static_cast<double>(
                        pixels[(y * width + x) * channels + c]
                    );
            }
        }
    }

    return out;
}

// Convert one sample from NCHW to interleaved HWC.
inline vector<double> toInterleaved(
    const Tensor& in,
    size_t n = 0
) {
    detail::requireValid(in, "toInterleaved");

    if (n >= in.shape[0]) {
        detail::fail("toInterleaved", "sample index out of range");
    }

    const size_t C = in.shape[1];
    const size_t H = in.shape[2];
    const size_t W = in.shape[3];

    vector<double> out(H * W * C);

    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            for (size_t c = 0; c < C; ++c) {
                out[(y * W + x) * C + c] =
                    in.data[detail::flat(in, n, c, y, x)];
            }
        }
    }

    return out;
}

// Reorder channels: output channel i = input channel order[i].
inline Tensor permuteChannels(
    const Tensor& in,
    const vector<size_t>& order
) {
    detail::check(in, "permuteChannels");

    const size_t N = in.shape[0];
    const size_t C = in.shape[1];
    const size_t H = in.shape[2];
    const size_t W = in.shape[3];

    if (order.size() != C) {
        detail::fail(
            "permuteChannels",
            "order size must equal channel count"
        );
    }

    vector<bool> seen(C, false);

    for (size_t src : order) {
        if (src >= C || seen[src]) {
            detail::fail(
                "permuteChannels",
                "order is not a permutation"
            );
        }

        seen[src] = true;
    }

    Tensor out(N, C, H, W);
    const size_t plane = H * W;

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const double* src =
                in.data.data() + detail::flat(in, n, order[c], 0, 0);

            double* dst =
                out.data.data() + detail::flat(out, n, c, 0, 0);

            copy(src, src + plane, dst);
        }
    }

    return out;
}

// Reverse channel order, for example RGB to BGR.
inline Tensor reverseChannels(const Tensor& in) {
    detail::requireValid(in, "reverseChannels");

    vector<size_t> order(in.shape[1]);

    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = order.size() - 1 - i;
    }

    return permuteChannels(in, order);
}

// Convert {N,3,H,W} RGB to {N,1,H,W} grayscale.
inline Tensor rgbToGrayscale(const Tensor& in) {
    detail::check(in, "rgbToGrayscale");

    if (in.shape[1] != 3) {
        detail::fail(
            "rgbToGrayscale",
            "input must have 3 channels"
        );
    }

    const size_t N = in.shape[0];
    const size_t H = in.shape[2];
    const size_t W = in.shape[3];
    const size_t plane = H * W;

    Tensor out(N, 1, H, W);

    for (size_t n = 0; n < N; ++n) {
        const size_t base = detail::flat(in, n, 0, 0, 0);

        for (size_t i = 0; i < plane; ++i) {
            out.data[n * plane + i] =
                0.299 * in.data[base + i] +
                0.587 * in.data[base + plane + i] +
                0.114 * in.data[base + 2 * plane + i];
        }
    }

    return out;
}

// Convert {N,1,H,W} to {N,3,H,W} by replicating the channel.
inline Tensor grayscaleToRgb(const Tensor& in) {
    detail::check(in, "grayscaleToRgb");

    if (in.shape[1] != 1) {
        detail::fail(
            "grayscaleToRgb",
            "input must have 1 channel"
        );
    }

    const size_t N = in.shape[0];
    const size_t H = in.shape[2];
    const size_t W = in.shape[3];
    const size_t plane = H * W;

    Tensor out(N, 3, H, W);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < 3; ++c) {
            copy(
                in.data.begin() + static_cast<ptrdiff_t>(n * plane),
                in.data.begin() + static_cast<ptrdiff_t>((n + 1) * plane),
                out.data.begin() +
                    static_cast<ptrdiff_t>((n * 3 + c) * plane)
            );
        }
    }

    return out;
}

// Convert to 1 or 3 channels.
inline Tensor toChannels(const Tensor& in, size_t target) {
    detail::requireValid(in, "toChannels");

    if (target != 1 && target != 3) {
        detail::fail("toChannels", "target must be 1 or 3");
    }

    const size_t C = in.shape[1];

    if (C == target) {
        return in;
    }

    if (C == 3 && target == 1) {
        return rgbToGrayscale(in);
    }

    if (C == 1 && target == 3) {
        return grayscaleToRgb(in);
    }

    detail::fail(
        "toChannels",
        "input must have 1 or 3 channels"
    );
}

// Resize the spatial dimensions using nearest-neighbor or bilinear
// interpolation. Bilinear interpolation uses half-pixel centers.
inline Tensor resize(
    const Tensor& in,
    size_t outH,
    size_t outW,
    Interpolation mode = Interpolation::Bilinear
) {
    detail::check(in, "resize");

    if (outH == 0 || outW == 0) {
        detail::fail("resize", "target size must be positive");
    }

    const size_t N = in.shape[0];
    const size_t C = in.shape[1];
    const size_t H = in.shape[2];
    const size_t W = in.shape[3];

    Tensor out = detail::makeOutput(
        N, C, outH, outW, "resize"
    );

    const double scaleY =
        static_cast<double>(H) / static_cast<double>(outH);

    const double scaleX =
        static_cast<double>(W) / static_cast<double>(outW);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const double* src =
                in.data.data() + detail::flat(in, n, c, 0, 0);

            double* dst =
                out.data.data() + detail::flat(out, n, c, 0, 0);

            for (size_t y = 0; y < outH; ++y) {
                if (mode == Interpolation::Nearest) {
                    const size_t sy = min(
                        H - 1,
                        static_cast<size_t>(
                            floor((static_cast<double>(y) + 0.5) * scaleY)
                        )
                    );

                    for (size_t x = 0; x < outW; ++x) {
                        const size_t sx = min(
                            W - 1,
                            static_cast<size_t>(
                                floor((static_cast<double>(x) + 0.5) * scaleX)
                            )
                        );

                        dst[y * outW + x] = src[sy * W + sx];
                    }
                } else {
                    double fy =
                        (static_cast<double>(y) + 0.5) * scaleY - 0.5;

                    fy = min(max(fy, 0.0), static_cast<double>(H - 1));

                    const size_t y0 = static_cast<size_t>(fy);
                    const size_t y1 = min(y0 + 1, H - 1);
                    const double wy = fy - static_cast<double>(y0);

                    for (size_t x = 0; x < outW; ++x) {
                        double fx =
                            (static_cast<double>(x) + 0.5) * scaleX - 0.5;

                        fx = min(max(fx, 0.0), static_cast<double>(W - 1));

                        const size_t x0 = static_cast<size_t>(fx);
                        const size_t x1 = min(x0 + 1, W - 1);
                        const double wx = fx - static_cast<double>(x0);

                        const double top =
                            src[y0 * W + x0] * (1.0 - wx) +
                            src[y0 * W + x1] * wx;

                        const double bottom =
                            src[y1 * W + x0] * (1.0 - wx) +
                            src[y1 * W + x1] * wx;

                        dst[y * outW + x] =
                            top * (1.0 - wy) + bottom * wy;
                    }
                }
            }
        }
    }

    return out;
}

// Scale values by maxValue. For 8-bit pixels, use 255.
inline Tensor scaleToUnit(
    const Tensor& in,
    double maxValue = 255.0
) {
    detail::check(in, "scaleToUnit");

    if (!(maxValue > 0.0) || !isfinite(maxValue)) {
        detail::fail(
            "scaleToUnit",
            "maxValue must be finite and > 0"
        );
    }

    Tensor out = in;

    for (double& v : out.data) {
        v /= maxValue;
    }

    return out;
}

// Standardize each channel using (x - mean) / stddev.
inline Tensor standardize(
    const Tensor& in,
    const vector<double>& mean,
    const vector<double>& stddev
) {
    detail::check(in, "standardize");

    const size_t N = in.shape[0];
    const size_t C = in.shape[1];
    const size_t plane = in.shape[2] * in.shape[3];

    if (mean.size() != C || stddev.size() != C) {
        detail::fail(
            "standardize",
            "mean/stddev size must equal channel count"
        );
    }

    for (size_t c = 0; c < C; ++c) {
        if (!isfinite(mean[c])) {
            detail::fail("standardize", "mean must be finite");
        }

        if (!(stddev[c] > 0.0) || !isfinite(stddev[c])) {
            detail::fail(
                "standardize",
                "stddev must be finite and > 0"
            );
        }
    }

    Tensor out = in;

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            double* p =
                out.data.data() + detail::flat(out, n, c, 0, 0);

            for (size_t i = 0; i < plane; ++i) {
                p[i] = (p[i] - mean[c]) / stddev[c];
            }
        }
    }

    return out;
}

// Per-sample min-max normalization. Constant samples map to lo.
inline Tensor minMaxNormalize(
    const Tensor& in,
    double lo = 0.0,
    double hi = 1.0
) {
    detail::check(in, "minMaxNormalize");

    if (!(lo < hi) || !isfinite(lo) || !isfinite(hi)) {
        detail::fail(
            "minMaxNormalize",
            "need finite lo < hi"
        );
    }

    const size_t N = in.shape[0];
    const size_t per =
        in.shape[1] * in.shape[2] * in.shape[3];

    Tensor out = in;

    for (size_t n = 0; n < N; ++n) {
        double* p = out.data.data() + n * per;

        const double mn = *min_element(p, p + per);
        const double mx = *max_element(p, p + per);

        for (size_t i = 0; i < per; ++i) {
            p[i] = (mx == mn)
                ? lo
                : lo + (p[i] - mn) * (hi - lo) / (mx - mn);
        }
    }

    return out;
}

// Compute per-channel population mean and standard deviation
// across N, H and W. Constant channels use stddev = 1.
inline ChannelStats computeChannelStats(const Tensor& in) {
    detail::check(in, "computeChannelStats");

    const size_t N = in.shape[0];
    const size_t C = in.shape[1];
    const size_t plane = in.shape[2] * in.shape[3];

    const double total = static_cast<double>(N * plane);

    ChannelStats stats;
    stats.mean.assign(C, 0.0);
    stats.stddev.assign(C, 1.0);

    for (size_t c = 0; c < C; ++c) {
        double sum = 0.0;

        for (size_t n = 0; n < N; ++n) {
            const double* p =
                in.data.data() + detail::flat(in, n, c, 0, 0);

            for (size_t i = 0; i < plane; ++i) {
                sum += p[i];
            }
        }

        const double avg = sum / total;
        double squaredSum = 0.0;

        for (size_t n = 0; n < N; ++n) {
            const double* p =
                in.data.data() + detail::flat(in, n, c, 0, 0);

            for (size_t i = 0; i < plane; ++i) {
                const double difference = p[i] - avg;
                squaredSum += difference * difference;
            }
        }

        const double sd = sqrt(squaredSum / total);

        stats.mean[c] = avg;
        stats.stddev[c] = (sd > 1e-12) ? sd : 1.0;
    }

    return stats;
}

// Concatenate tensors along the batch dimension.
// Every tensor must have matching C, H and W.
inline Tensor stackBatch(const vector<Tensor>& items) {
    if (items.empty()) {
        detail::fail("stackBatch", "empty list");
    }

    size_t totalN = 0;

    for (const Tensor& t : items) {
        detail::requireValid(t, "stackBatch");

        if (t.shape[1] != items[0].shape[1] ||
            t.shape[2] != items[0].shape[2] ||
            t.shape[3] != items[0].shape[3]) {
            detail::fail(
                "stackBatch",
                "all tensors must share C, H, W"
            );
        }

        if (t.shape[0] > numeric_limits<size_t>::max() - totalN) {
            detail::fail("stackBatch", "batch too large");
        }

        totalN += t.shape[0];
    }

    Tensor out = detail::makeOutput(
        totalN,
        items[0].shape[1],
        items[0].shape[2],
        items[0].shape[3],
        "stackBatch"
    );

    size_t offset = 0;

    for (const Tensor& t : items) {
        copy(
            t.data.begin(),
            t.data.end(),
            out.data.begin() + static_cast<ptrdiff_t>(offset)
        );

        offset += t.data.size();
    }

    return out;
}

// Validate the preprocessing configuration.
inline void validateConfig(const PreprocessConfig& cfg) {
    if ((cfg.target_height == 0) != (cfg.target_width == 0)) {
        detail::fail(
            "validateConfig",
            "set both target_height and target_width, or neither"
        );
    }

    if (cfg.target_channels != 0 &&
        cfg.target_channels != 1 &&
        cfg.target_channels != 3) {
        detail::fail(
            "validateConfig",
            "target_channels must be 0, 1 or 3"
        );
    }

    if ((cfg.normalization == Normalization::Scale ||
         cfg.normalization == Normalization::Standardize) &&
        (!(cfg.max_value > 0.0) || !isfinite(cfg.max_value))) {
        detail::fail(
            "validateConfig",
            "max_value must be finite and > 0"
        );
    }

    if (cfg.normalization == Normalization::MinMax &&
        (!(cfg.range_min < cfg.range_max) ||
         !isfinite(cfg.range_min) ||
         !isfinite(cfg.range_max))) {
        detail::fail(
            "validateConfig",
            "need finite range_min < range_max"
        );
    }
}

// Full preprocessing pipeline.
// Input and output use {N,C,H,W} format.
// Standardize first scales values, then applies (x - mean) / stddev.
inline Tensor preprocess(
    const Tensor& input,
    const PreprocessConfig& cfg
) {
    validateConfig(cfg);
    detail::check(input, "preprocess");

    Tensor cur = input;

    // Use RGB weights for grayscale conversion before channel reversal.
    if (cfg.target_channels == 1 && cur.shape[1] == 3) {
        cur = rgbToGrayscale(cur);
    } else if (cfg.reverse_channels) {
        cur = reverseChannels(cur);
    }

    if (cfg.target_channels != 0 &&
        cur.shape[1] != cfg.target_channels) {
        cur = toChannels(cur, cfg.target_channels);
    }

    if (cfg.target_height != 0) {
        cur = resize(
            cur,
            cfg.target_height,
            cfg.target_width,
            cfg.interpolation
        );
    }

    switch (cfg.normalization) {
        case Normalization::None:
            break;

        case Normalization::Scale:
            cur = scaleToUnit(cur, cfg.max_value);
            break;

        case Normalization::Standardize:
            cur = standardize(
                scaleToUnit(cur, cfg.max_value),
                cfg.mean,
                cfg.stddev
            );
            break;

        case Normalization::MinMax:
            cur = minMaxNormalize(
                cur,
                cfg.range_min,
                cfg.range_max
            );
            break;
    }

    return cur;
}

} // namespace cnn

#endif // CNN_PREPROCESS_HPP
