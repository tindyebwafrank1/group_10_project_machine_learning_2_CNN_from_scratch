#ifndef POOLING_BACKWARD_HPP
#define POOLING_BACKWARD_HPP

#include "tensor.hpp"
#include <vector>
#include <cstddef>

using namespace std;

class PoolingBackward {
public:
    size_t pool_height;
    size_t pool_width;
    size_t stride;

    // Stores flat 1D coordinates of max elements filled during forward pass
    vector<size_t> max_indices;

    PoolingBackward(size_t p_height, size_t p_width, size_t s_stride);

    // Max pooling backward pass: Routes gradients using forward index tracking
    Tensor max_pooling_backward(const Tensor& dL_dy, const vector<size_t>& input_shape);

    // Average pooling backward pass: Distributes gradients equally over the pool area
    Tensor average_pooling_backward(const Tensor& dL_dy, const vector<size_t>& input_shape);
};

#endif 
