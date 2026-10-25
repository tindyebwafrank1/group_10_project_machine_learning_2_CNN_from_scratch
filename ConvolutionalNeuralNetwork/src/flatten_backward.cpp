#include "flatten_backward.hpp"

#include <stdexcept>
#include <vector>

using namespace std;

namespace cnn {

// Restore a flattened gradient to the original tensor shape.
Tensor unflattenGradient(
    const vector<double>& flattened_gradient,
    const Tensor& original_input)
{
    // The original input must have four dimensions: N, C, H, W.
    if (original_input.shape.size() != 4) {
        throw invalid_argument(
            "Original input must be a 4D tensor");
    }

    // The gradient must contain one value for every input element.
    if (flattened_gradient.size() != original_input.size()) {
        throw invalid_argument(
            "Gradient size must match the original tensor size");
    }

    // Create a tensor with the same dimensions as the original input.
    Tensor gradient(
        original_input.shape[0],
        original_input.shape[1],
        original_input.shape[2],
        original_input.shape[3]
    );

    // Copy the gradient values back in the same contiguous order.
    // This restores the original shape without changing the values.
    gradient.data = flattened_gradient;

    return gradient;
}

} 