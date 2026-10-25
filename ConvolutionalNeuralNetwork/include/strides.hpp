
#ifndef CNN_STRIDES_HPP
#define CNN_STRIDES_HPP

#include <cstddef>

#include "tensor.hpp"

using namespace std;

namespace cnn {

// Calculates the output dimension for a convolution or pooling operation.
// Formula: floor((input + 2*padding - kernel) / stride) + 1.
size_t calculateStridedOutputSize(
    size_t input_size,
    size_t kernel_size,
    size_t stride,
    size_t padding = 0
);

// Samples a 4-D NCHW tensor by selecting every stride_h-th row
// and every stride_w-th column, starting at position (0,0).
Tensor subsample(
    const Tensor& input,
    size_t stride_h,
    size_t stride_w
);

// Convenience overload for equal horizontal and vertical strides.
Tensor subsample(const Tensor& input, size_t stride);

} // namespace cnn

#endif
