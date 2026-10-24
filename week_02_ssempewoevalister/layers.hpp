#ifndef LAYERS_HPP
#define LAYERS_HPP
// STAND-IN for Joan/Evarister's Conv2D (Week 1) with the padding fix applied.
// Delete once the real layers.hpp (with padding support from Frank, stride from Peter) is on main.
#include <stdexcept>
#include "cnn_library/tensor.hpp"
namespace CNN {
    class Conv2D {
        size_t stride, padding;
    public:
        Conv2D(size_t s = 1, size_t p = 0) : stride(s), padding(p) {}
        Tensor forward(const Tensor& input, const Tensor& kernel) const {
            size_t N = input.shape[0], C = input.shape[1], H = input.shape[2], W = input.shape[3];
            size_t K = kernel.shape[0], Kh = kernel.shape[2], Kw = kernel.shape[3];
            if (kernel.shape[1] != C) throw std::invalid_argument("Kernel channels must match input channels");
            if (stride == 0) throw std::invalid_argument("Stride must be > 0");
            if (H + 2 * padding < Kh || W + 2 * padding < Kw) throw std::invalid_argument("Kernel larger than padded input");
            size_t out_H = (H + 2 * padding - Kh) / stride + 1, out_W = (W + 2 * padding - Kw) / stride + 1;
            Tensor out(N, K, out_H, out_W);
            for (size_t n = 0; n < N; ++n) for (size_t k = 0; k < K; ++k)
              for (size_t h = 0; h < out_H; ++h) for (size_t w = 0; w < out_W; ++w) {
                double acc = 0.0;
                for (size_t c = 0; c < C; ++c) for (size_t r = 0; r < Kh; ++r) for (size_t s = 0; s < Kw; ++s) {
                    long ih = (long)(h * stride + r) - (long)padding, iw = (long)(w * stride + s) - (long)padding;
                    if (ih < 0 || ih >= (long)H || iw < 0 || iw >= (long)W) continue;
                    acc += input(n, c, ih, iw) * kernel(k, c, r, s);
                }
                out(n, k, h, w) = acc;
            }
            return out;
        }
    };
}
#endif
