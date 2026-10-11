#ifndef CNN_CONVOLUTION_HPP
#define CNN_CONVOLUTION_HPP

#include <cstddef>
#include "cnn/tensor.hpp"   // Agatha's Tensor (M4)

namespace cnn {

// Output size along one axis, no padding, stride 1:
//   out = in - kernel + 1
// Throws std::invalid_argument if kernel == 0 or kernel > in.
std::size_t conv_output_dim(std::size_t in, std::size_t kernel);

// Dot product of the kernel with the window of the input whose
// top-left corner is (h, w). Single channel: both tensors are (1,1,H,W).
double window_dot(const Tensor& input, const Tensor& kernel,
                  std::size_t h, std::size_t w);

// Valid 2D convolution (cross-correlation) of a (1,1,H,W) input with a
// (1,1,Kh,Kw) kernel. Returns a (1,1,H-Kh+1,W-Kw+1) tensor.
// Throws std::invalid_argument on wrong shapes or a kernel larger than the input.
Tensor conv2d_single(const Tensor& input, const Tensor& kernel);

}  // namespace cnn

#endif  // CNN_CONVOLUTION_HPP