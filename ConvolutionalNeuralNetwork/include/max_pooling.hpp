
#ifndef CNN_MAX_POOLING_HPP
#define CNN_MAX_POOLING_HPP

#include "tensor.hpp"

#include <cstddef>
#include <vector>

using namespace std;

// Defines the pooling window and movement between windows.
struct PoolingParams
{
    size_t window_h = 2;
    size_t window_w = 2;

    size_t stride_h = 2;
    size_t stride_w = 2;
};

// Stores the pooled output and the input positions of its maxima.
struct MaxPoolingResult
{
    Tensor output;

    // Each entry is the flat index into the original input Tensor::data.
    // Entries correspond to output.data in the same order.
    vector<size_t> max_indices;
};

// Applies 2D max pooling to a 4D NCHW input tensor.
MaxPoolingResult maxPool2D(
    const Tensor& input,
    const PoolingParams& params = PoolingParams{}
);

#endif
