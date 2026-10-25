#ifndef CNN_FLATTEN_BACKWARD_HPP
#define CNN_FLATTEN_BACKWARD_HPP

#include <vector>
#include "tensor.hpp"

using namespace std;

namespace cnn {

// Routes a flattened gradient back into the original 4D tensor shape.
Tensor unflattenGradient(
    const vector<double>& flattened_gradient,
    const Tensor& original_input
);

} // namespace cnn

#endif