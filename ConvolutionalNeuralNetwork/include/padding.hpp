
#ifndef CNN_PADDING_HPP
#define CNN_PADDING_HPP

#include "tensor.hpp"
#include <cstddef>

namespace cnn {

enum class PaddingMode {
    Valid,
    Same
};

// Add zeros around the spatial dimensions of a tensor.
Tensor zeroPad(const Tensor& input, size_t pad_h, size_t pad_w);

// Calculate the padding needed for Same mode.
void calculateSamePadding(
    size_t input_h,
    size_t input_w,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t& pad_top,
    size_t& pad_bottom,
    size_t& pad_left,
    size_t& pad_right
);

} // namespace cnn

#endif
