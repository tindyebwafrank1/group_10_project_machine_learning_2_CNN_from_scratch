#include "cnn/multichannel_conv.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace cnn {

std::size_t conv_output_size(std::size_t in, std::size_t k) {
    if (k == 0) throw std::invalid_argument("conv_output_size: kernel size must be > 0");
    if (k > in) throw std::invalid_argument("conv_output_size: kernel larger than input");
    return in - k + 1;
}

namespace {
void check_tensor(const Tensor& t, const char* name) {
    if (t.shape.size() != 4)
        throw std::invalid_argument(std::string(name) + ": tensor must be 4D (N,C,H,W)");
    const std::size_t expected = t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3];
    if (t.data.size() != expected)
        throw std::invalid_argument(std::string(name) + ": data size does not match shape");
    if (expected == 0)
        throw std::invalid_argument(std::string(name) + ": tensor is empty");
}
}  // namespace

Tensor conv2d_multichannel(const Tensor& input, const Tensor& filters) {
    check_tensor(input, "input");
    check_tensor(filters, "filters");

    const std::size_t N = input.shape[0], C = input.shape[1];
    const std::size_t H = input.shape[2], W = input.shape[3];
    const std::size_t K = filters.shape[0];
    const std::size_t Kh = filters.shape[2], Kw = filters.shape[3];

    if (filters.shape[1] != C)
        throw std::invalid_argument("conv2d_multichannel: filter channels (" +
                                    std::to_string(filters.shape[1]) +
                                    ") != input channels (" + std::to_string(C) + ")");

    const std::size_t Ho = conv_output_size(H, Kh);
    const std::size_t Wo = conv_output_size(W, Kw);

    Tensor out(N, K, Ho, Wo);
    std::fill(out.data.begin(), out.data.end(), 0.0);

    const double* x = input.data.data();
    const double* f = filters.data.data();
    double* y = out.data.data();

    // Flattened NCHW indexing: idx(n,c,h,w) = ((n*C + c)*H + h)*W + w
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t h = 0; h < Ho; ++h) {
                for (std::size_t w = 0; w < Wo; ++w) {
                    double acc = 0.0;
                    for (std::size_t c = 0; c < C; ++c) {
                        for (std::size_t r = 0; r < Kh; ++r) {
                            const double* xrow = x + ((n * C + c) * H + (h + r)) * W + w;
                            const double* frow = f + ((k * C + c) * Kh + r) * Kw;
                            for (std::size_t s = 0; s < Kw; ++s) acc += xrow[s] * frow[s];
                        }
                    }
                    y[((n * K + k) * Ho + h) * Wo + w] = acc;
                }
            }
        }
    }
    return out;
}

}  // namespace cnn
