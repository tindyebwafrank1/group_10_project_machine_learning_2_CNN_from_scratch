
#ifndef RELU_HPP
#define RELU_HPP

#include "tensor.hpp"

namespace cnn {

// Apply ReLU to a tensor and return the result.
Tensor relu(const Tensor& input);

// Apply ReLU directly to an existing tensor.
void reluInPlace(Tensor& input);

} 

#endif
