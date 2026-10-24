// multichannel_conv.hpp
// Week 1 | Module  (Ssempewo evalister): multi-channel / multi-filter convolution pass.
//
// Depends on: Tensor ( Agatha). Layout is NCHW in one contiguous vector<double>.
#ifndef CNN_MULTICHANNEL_CONV_HPP
#define CNN_MULTICHANNEL_CONV_HPP

#include <cstddef>
#include "cnn/tensor.hpp"

namespace cnn {

// Output length along one spatial axis for a "valid" convolution
// (no padding, stride 1):  out = in - k + 1.
// Throws std::invalid_argument if k == 0 or k > in.
std::size_t conv_output_size(std::size_t in, std::size_t k);

// Convolves a batch of multi-channel images with a bank of multi-filter kernels.
//
//   input   : shape {N, C,  H,  W}
//   filters : shape {K, C, Kh, Kw}   (K filters, each spanning all C channels)
//   returns : shape {N, K, H-Kh+1, W-Kw+1}
//
//   Y(n,k,h,w) = sum_c sum_r sum_s X(n,c,h+r,w+s) * F(k,c,r,s)
//
// Every filter sums its contribution over ALL input channels, so each filter
// yields exactly one feature map per image.
//
// Throws std::invalid_argument if either tensor is not 4D, the channel counts
// differ, the data size does not match the shape, or the kernel is larger than
// the input.
Tensor conv2d_multichannel(const Tensor& input, const Tensor& filters);

}  // namespace cnn

#endif  // CNN_MULTICHANNEL_CONV_HPP
