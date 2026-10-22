#include "pooling_backward.hpp"
#include <stdexcept>

using namespace std;

PoolingBackward::PoolingBackward(size_t p_height, size_t p_width, size_t s_stride)
    : pool_height(p_height), pool_width(p_width), stride(s_stride) {}

Tensor PoolingBackward::max_pooling_backward(const Tensor& dL_dy, const vector<size_t>& input_shape) {
    if (input_shape.size() != 4) {
        throw invalid_argument("max_pooling_backward: Input shape must be 4D {N, C, H, W}.");
    }

    // 1. Allocate a zero-filled gradient tensor matching the original input shape
    Tensor dL_dx(input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

    size_t batch = dL_dy.shape[0];
    size_t channels = dL_dy.shape[1];
    size_t out_h = dL_dy.shape[2];
    size_t out_w = dL_dy.shape[3];

    // 2. Map upstream gradients back to original max element coordinates
    for (size_t n = 0; n < batch; ++n) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    
                    // Locate upstream gradient flat position
                    size_t out_flat_idx = n * (channels * out_h * out_w)
                                        + c * (out_h * out_w)
                                        + oh * out_w
                                        + ow;

                    // Fetch the original max index tracked during Week 2 forward pass
                    size_t target_input_flat_idx = max_indices[out_flat_idx];

                    // Accumulate gradient into the tracked max pixel slot (handles overlaps cleanly)
                    dL_dx.data[target_input_flat_idx] += dL_dy(n, c, oh, ow);
                }
            }
        }
    }

    return dL_dx;
}

Tensor PoolingBackward::average_pooling_backward(const Tensor& dL_dy, const vector<size_t>& input_shape) {
    if (input_shape.size() != 4) {
        throw invalid_argument("average_pooling_backward: Input shape must be 4D {N, C, H, W}.");
    }

    Tensor dL_dx(input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

    size_t batch = dL_dy.shape[0];
    size_t channels = dL_dy.shape[1];
    size_t out_h = dL_dy.shape[2];
    size_t out_w = dL_dy.shape[3];

    size_t in_h = input_shape[2];
    size_t in_w = input_shape[3];

    // Scale coefficient factor for uniform average distribution
    double scale = 1.0 / static_cast<double>(pool_height * pool_width);

    // 3. Loop over the output spatial dimensions to distribute the average gradient
    for (size_t n = 0; n < batch; ++n) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    
                    double upstream_grad = dL_dy(n, c, oh, ow);
                    size_t start_h = oh * stride;
                    size_t start_w = ow * stride;

                    // Distribute uniformly across the pooling window elements
                    for (size_t kh = 0; kh < pool_height; ++kh) {
                        for (size_t kw = 0; kw < pool_width; ++kw) {
                            size_t curr_h = start_h + kh;
                            size_t curr_w = start_w + kw;

                            if (curr_h < in_h && curr_w < in_w) {
                                dL_dx(n, c, curr_h, curr_w) += upstream_grad * scale;
                            }
                        }
                    }

                }
            }
        }
    }

    return dL_dx;
}
