
#ifndef CNN_MULTI_CHANNEL_CONVOLUTION_HPP
#define CNN_MULTI_CHANNEL_CONVOLUTION_HPP

#include "tensor.hpp"

using namespace std;

namespace cnn {

// Multi-channel, multi-filter convolution.
//
// Input shape:   (N, C_in, H, W)
// Filters shape: (C_out, C_in, K_h, K_w)
// Output shape:  (N, C_out, H_out, W_out)
//
// Uses stride = 1 and no padding.
Tensor multiChannelConvolution(
    const Tensor& input,
    const Tensor& filters
);

} // namespace cnn

#endif
