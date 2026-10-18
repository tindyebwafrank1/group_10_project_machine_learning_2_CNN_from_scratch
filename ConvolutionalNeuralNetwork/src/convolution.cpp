#include "cnn/convolution.hpp"
#include <stdexcept>
#include <string>

namespace cnn {

std::size_t conv_output_dim(std::size_t in, std::size_t kernel) {
    if (kernel == 0)
        throw std::invalid_argument("conv_output_dim: kernel must be > 0");
    if (kernel > in)
        throw std::invalid_argument("conv_output_dim: kernel larger than input");
    return in - kernel + 1;
}

static void check_single_channel(const Tensor& t, const char* name) {
    if (t.shape.size() != 4 || t.shape[0] != 1 || t.shape[1] != 1)
        throw std::invalid_argument(std::string(name) + " must have shape (1,1,H,W)");
}

double window_dot(const Tensor& input, const Tensor& kernel,
                  std::size_t h, std::size_t w) {
    const std::size_t kh = kernel.shape[2], kw = kernel.shape[3];
    double sum = 0.0;
    for (std::size_t r = 0; r < kh; ++r)
        for (std::size_t s = 0; s < kw; ++s)
            sum += input(0, 0, h + r, w + s) * kernel(0, 0, r, s);
    return sum;
}

Tensor conv2d_single(const Tensor& input, const Tensor& kernel) {
    check_single_channel(input, "input");
    check_single_channel(kernel, "kernel");

    const std::size_t out_h = conv_output_dim(input.shape[2], kernel.shape[2]);
    const std::size_t out_w = conv_output_dim(input.shape[3], kernel.shape[3]);

    Tensor out(1, 1, out_h, out_w);
    for (std::size_t h = 0; h < out_h; ++h)
        for (std::size_t w = 0; w < out_w; ++w)
            out(0, 0, h, w) = window_dot(input, kernel, h, w);
    return out;
}

}  // namespace cnn