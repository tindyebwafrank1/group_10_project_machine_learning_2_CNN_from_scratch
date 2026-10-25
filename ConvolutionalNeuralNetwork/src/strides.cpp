
#include "strides.hpp"

#include <limits>
#include <stdexcept>

using namespace std;

namespace cnn {

namespace {

void validateTensor(const Tensor& input)
{
    if (input.shape.size() != 4) {
        throw invalid_argument("Input tensor must be 4-D (N,C,H,W)");
    }

    const size_t N = input.shape[0];
    const size_t C = input.shape[1];
    const size_t H = input.shape[2];
    const size_t W = input.shape[3];

    if (N == 0 || C == 0 || H == 0 || W == 0) {
        throw invalid_argument("Tensor dimensions must be positive");
    }

    if (N > numeric_limits<size_t>::max() / C ||
        N * C > numeric_limits<size_t>::max() / H ||
        N * C * H > numeric_limits<size_t>::max() / W) {
        throw overflow_error("Tensor dimensions are too large");
    }

    if (input.data.size() != N * C * H * W) {
        throw invalid_argument("Tensor data size does not match its shape");
    }
}

} // namespace


size_t calculateStridedOutputSize(
    size_t input_size,
    size_t kernel_size,
    size_t stride,
    size_t padding)
{
    if (input_size == 0 || kernel_size == 0 || stride == 0) {
        throw invalid_argument(
            "Input size, kernel size, and stride must be positive");
    }

    if (padding > (numeric_limits<size_t>::max() - input_size) / 2) {
        throw overflow_error("Padded input size is too large");
    }

    const size_t padded_size = input_size + 2 * padding;

    if (kernel_size > padded_size) {
        throw invalid_argument(
            "Kernel size cannot exceed the padded input size");
    }

    return (padded_size - kernel_size) / stride + 1;
}


Tensor subsample(
    const Tensor& input,
    size_t stride_h,
    size_t stride_w)
{
    validateTensor(input);

    if (stride_h == 0 || stride_w == 0) {
        throw invalid_argument("Stride values must be positive");
    }

    const size_t N = input.shape[0];
    const size_t C = input.shape[1];
    const size_t H = input.shape[2];
    const size_t W = input.shape[3];

    // Sample positions 0, stride, 2*stride, and so on.
    // This gives ceil(H / stride_h) rows and ceil(W / stride_w) columns.
    const size_t output_height = (H - 1) / stride_h + 1;
    const size_t output_width = (W - 1) / stride_w + 1;

    Tensor output(N, C, output_height, output_width);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t out_h = 0; out_h < output_height; ++out_h) {
                for (size_t out_w = 0; out_w < output_width; ++out_w) {

                    const size_t in_h = out_h * stride_h;
                    const size_t in_w = out_w * stride_w;

                    output(n, c, out_h, out_w) =
                        input(n, c, in_h, in_w);
                }
            }
        }
    }

    return output;
}


Tensor subsample(const Tensor& input, size_t stride)
{
    return subsample(input, stride, stride);
}

} // namespace cnn
