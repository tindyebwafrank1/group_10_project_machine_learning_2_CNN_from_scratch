
#include "padding.hpp"

#include <limits>
#include <stdexcept>

namespace cnn {

Tensor zeroPad(const Tensor& input, size_t pad_h, size_t pad_w)
{
    if (input.shape.size() != 4) {
        throw std::invalid_argument("Input tensor must be 4-D");
    }

    const size_t N = input.shape[0];
    const size_t C = input.shape[1];
    const size_t H = input.shape[2];
    const size_t W = input.shape[3];

    if (N == 0 || C == 0 || H == 0 || W == 0) {
        throw std::invalid_argument("Tensor dimensions must be positive");
    }

    if (input.size() != input.data.size()) {
        throw std::invalid_argument("Invalid tensor data size");
    }

    if (pad_h > (std::numeric_limits<size_t>::max() - H) / 2 ||
        pad_w > (std::numeric_limits<size_t>::max() - W) / 2) {
        throw std::overflow_error("Padding size is too large");
    }

    const size_t new_H = H + 2 * pad_h;
    const size_t new_W = W + 2 * pad_w;

    Tensor output(N, C, new_H, new_W);

    // Initialize every output element to zero.
    output.fill(0.0);

    // Copy the original values into the center.
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    output(n, c, h + pad_h, w + pad_w) =
                        input(n, c, h, w);
                }
            }
        }
    }

    return output;
}

void calculateSamePadding(
    size_t input_h,
    size_t input_w,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t& pad_top,
    size_t& pad_bottom,
    size_t& pad_left,
    size_t& pad_right)
{
    if (input_h == 0 || input_w == 0 ||
        kernel_h == 0 || kernel_w == 0 ||
        stride_h == 0 || stride_w == 0) {
        throw std::invalid_argument(
            "Input, kernel and stride dimensions must be positive");
    }

    // Calculate the output dimensions for Same mode.
    const size_t out_h = input_h / stride_h +
                         (input_h % stride_h != 0);
    const size_t out_w = input_w / stride_w +
                         (input_w % stride_w != 0);

    // Calculate the total padding required.
    const size_t needed_h =
        out_h > 0 && out_h - 1 >
        (std::numeric_limits<size_t>::max() - kernel_h) / stride_h
        ? throw std::overflow_error("Height padding overflow"),
          size_t(0)
        : (out_h - 1) * stride_h + kernel_h;

    const size_t needed_w =
        out_w > 0 && out_w - 1 >
        (std::numeric_limits<size_t>::max() - kernel_w) / stride_w
        ? throw std::overflow_error("Width padding overflow"),
          size_t(0)
        : (out_w - 1) * stride_w + kernel_w;

    const size_t total_h =
        needed_h > input_h ? needed_h - input_h : 0;
    const size_t total_w =
        needed_w > input_w ? needed_w - input_w : 0;

    pad_top = total_h / 2;
    pad_bottom = total_h - pad_top;

    pad_left = total_w / 2;
    pad_right = total_w - pad_left;
}

} // namespace cnn
