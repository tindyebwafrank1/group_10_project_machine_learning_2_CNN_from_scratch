
#ifndef CNN_POOLING_BACKWARD_HPP
#define CNN_POOLING_BACKWARD_HPP

#include "max_pooling.hpp"

#include "tensor.hpp"

using namespace std;

// Distributes gradients through max pooling using saved forward indices.
// input is the original input tensor.
// forward_result is the result returned by maxPool2D().
// upstream_gradient contains gradients of the pooled output.
Tensor maxPool2DBackward(
    const Tensor& input,
    const MaxPoolingResult& forward_result,
    const Tensor& upstream_gradient
);

// Distributes gradients through average pooling.
// Each gradient is divided equally among the elements in its window.
Tensor averagePool2DBackward(
    const Tensor& input,
    const PoolingParams& params,
    const Tensor& upstream_gradient
);

#endif
