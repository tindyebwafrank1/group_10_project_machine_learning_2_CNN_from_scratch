#ifndef CONV_BACKWARD_HPP
#define CONV_BACKWARD_HPP

#include <cstddef>
#include <utility>
#include "tensor.hpp"

namespace cnn {

// Output size along one axis with zero padding and stride:
//   out = floor((in + 2*pad - kernel) / stride) + 1
// Throws std::invalid_argument if kernel == 0, stride == 0,
// or the kernel does not fit in the padded input.
std::size_t conv_out_size(std::size_t in, std::size_t kernel,
                          std::size_t stride, std::size_t pad);

// Gradients returned by the backward pass.
struct ConvGradients {
    Tensor input_grad;    // dL/dX, shape (N, C, H, W), same as the input
    Tensor kernel_grad;   // dL/dK, shape (K, C, Kh, Kw), same as the kernel

    ConvGradients(Tensor dx, Tensor dk)
        : input_grad(std::move(dx)), kernel_grad(std::move(dk)) {}
};

// Backward pass of Conv2D.
//   input:       (N, C, H, W)   the tensor given to the forward pass
//   kernel:      (K, C, Kh, Kw)
//   grad_output: (N, K, Ho, Wo) dL/dY coming from the next layer
// Throws std::invalid_argument on any shape mismatch.
ConvGradients conv2d_backward(const Tensor& input,
                              const Tensor& kernel,
                              const Tensor& grad_output,
                              std::size_t stride = 1,
                              std::size_t pad = 0);

}  // namespace cnn

#endif  //CONV_BACKWARD_HPP