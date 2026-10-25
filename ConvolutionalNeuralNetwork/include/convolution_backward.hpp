
#ifndef CNN_CONVOLUTION_BACKWARD_HPP
#define CNN_CONVOLUTION_BACKWARD_HPP

#include "tensor.hpp"

namespace cnn {

// Calculate the gradient of the loss with respect to the
// convolution kernels.
//
// input:         (N, Cin, H, W)
// kernels:       (Cout, Cin, Kh, Kw)
// grad_output:   (N, Cout, H-Kh+1, W-Kw+1)
//
// Returns a tensor with the same shape as kernels.
::Tensor convolutionKernelGradient(
    const ::Tensor& input,
    const ::Tensor& kernels,
    const ::Tensor& grad_output
);

// Calculate the gradient of the loss with respect to the
// input tensor.
//
// input:         (N, Cin, H, W)
// kernels:       (Cout, Cin, Kh, Kw)
// grad_output:   (N, Cout, H-Kh+1, W-Kw+1)
//
// Returns a tensor with the same shape as input.
::Tensor convolutionInputGradient(
    const ::Tensor& input,
    const ::Tensor& kernels,
    const ::Tensor& grad_output
);

} // namespace cnn

#endif // CNN_CONVOLUTION_BACKWARD_HPP
