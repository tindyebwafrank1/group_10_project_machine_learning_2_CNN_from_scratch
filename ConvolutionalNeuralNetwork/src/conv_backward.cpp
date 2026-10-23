#include "conv_backward.hpp"
#include <stdexcept>

namespace cnn {

std::size_t conv_out_size(std::size_t in, std::size_t kernel,
                          std::size_t stride, std::size_t pad) {
    if (kernel == 0)
        throw std::invalid_argument("conv_out_size: kernel must be > 0");
    if (stride == 0)
        throw std::invalid_argument("conv_out_size: stride must be > 0");
    const std::size_t padded = in + 2 * pad;
    if (kernel > padded)
        throw std::invalid_argument("conv_out_size: kernel larger than padded input");
    return (padded - kernel) / stride + 1;   // unsigned division floors
}

ConvGradients conv2d_backward(const Tensor& input,
                              const Tensor& kernel,
                              const Tensor& grad_output,
                              std::size_t stride,
                              std::size_t pad) {
    if (input.shape.size() != 4 || kernel.shape.size() != 4 ||
        grad_output.shape.size() != 4)
        throw std::invalid_argument("conv2d_backward: all tensors must be 4D");

    const std::size_t N = input.shape[0], C = input.shape[1];
    const std::size_t H = input.shape[2], W = input.shape[3];
    const std::size_t K = kernel.shape[0];
    const std::size_t Kh = kernel.shape[2], Kw = kernel.shape[3];

    if (kernel.shape[1] != C)
        throw std::invalid_argument("conv2d_backward: kernel channels != input channels");

    const std::size_t Ho = conv_out_size(H, Kh, stride, pad);
    const std::size_t Wo = conv_out_size(W, Kw, stride, pad);

    if (grad_output.shape[0] != N || grad_output.shape[1] != K ||
        grad_output.shape[2] != Ho || grad_output.shape[3] != Wo)
        throw std::invalid_argument("conv2d_backward: grad_output shape does not match forward output");

    Tensor dx(N, C, H, W);      // assumed zero-initialised by Tensor
    Tensor dk(K, C, Kh, Kw);

    const long p = static_cast<long>(pad);

    for (std::size_t n = 0; n < N; ++n)
      for (std::size_t k = 0; k < K; ++k)
        for (std::size_t i = 0; i < Ho; ++i)
          for (std::size_t j = 0; j < Wo; ++j) {
              const double g = grad_output(n, k, i, j);
              for (std::size_t c = 0; c < C; ++c)
                for (std::size_t r = 0; r < Kh; ++r)
                  for (std::size_t s = 0; s < Kw; ++s) {
                      // position in the unpadded input
                      const long h = static_cast<long>(i * stride + r) - p;
                      const long w = static_cast<long>(j * stride + s) - p;
                      if (h < 0 || h >= static_cast<long>(H) ||
                          w < 0 || w >= static_cast<long>(W))
                          continue;   // landed in the zero padding
                      const std::size_t hh = static_cast<std::size_t>(h);
                      const std::size_t ww = static_cast<std::size_t>(w);
                      dk(k, c, r, s)   += g * input(n, c, hh, ww);
                      dx(n, c, hh, ww) += g * kernel(k, c, r, s);
                  }
          }

    return ConvGradients(std::move(dx), std::move(dk));
}

}  // namespace cnn