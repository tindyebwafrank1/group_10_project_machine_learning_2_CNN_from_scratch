
#include "relu.hpp"

#include <algorithm>

namespace cnn {

Tensor relu(const Tensor& input)
{
    Tensor output = input;

    for (double& value : output.data) {
        value = std::max(0.0, value);
    }

    return output;
}

void reluInPlace(Tensor& input)
{
    for (double& value : input.data) {
        value = std::max(0.0, value);
    }
}

} 
