#include "max_pooling.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

MaxPooling::MaxPooling(size_t p_height, size_t p_width, size_t s_stride)
    : pool_height(p_height), pool_width(p_width), stride(s_stride) {}

Tensor MaxPooling::forward(const Tensor& input) {

    size_t batch = input.shape[0];
    size_t channels = input.shape[1];
    size_t in_h = input.shape[2];
    size_t in_w = input.shape[3];

    // Compute spatial dimensions using the project manual downsampling equation
    size_t out_h = (in_h - pool_height) / stride + 1;
    size_t out_w = (in_w - pool_width) / stride + 1;

    //  Allocate the condensed output tensor and prepare the max index tracking payload
    Tensor output(batch, channels, out_h, out_w);
    max_indices.resize(batch * channels * out_h * out_w, 0);

    //  Perform the Max Pooling sliding window downsampling operation
    for (size_t n = 0; n < batch; ++n) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                
                    size_t start_h = oh * stride;
                    size_t start_w = ow * stride;

                    double max_val = -std::numeric_limits<double>::infinity();
                    size_t max_flat_idx = 0;

                    for (size_t kh = 0; kh < pool_height; ++kh) {
                        for (size_t kw = 0; kw < pool_width; ++kw) {
                            size_t curr_h = start_h + kh;
                            size_t curr_w = start_w + kw;

                            // Edge case guard against spatial boundaries
                            if (curr_h < in_h && curr_w < in_w) {
                                
                                double val = input(n, c, curr_h, curr_w);
                                
                                if (val > max_val) {
                                    max_val = val;
                                    // Derive the raw 1D memory array address for backpropagation
                                    max_flat_idx = n * (input.shape[1] * input.shape[2] * input.shape[3])
                                                 + c * (input.shape[2] * input.shape[3])
                                                 + curr_h * input.shape[3]
                                                 + curr_w;
                                }
                            }
                        }
                    }

                                       output(n, c, oh, ow) = max_val;

                    size_t out_flat_idx = n * (channels * out_h * out_w)
                                        + c * (out_h * out_w)
                                        + oh * out_w
                                        + ow;
                    max_indices[out_flat_idx] = max_flat_idx;
                }
            }
        }
    }

    return output;
}
