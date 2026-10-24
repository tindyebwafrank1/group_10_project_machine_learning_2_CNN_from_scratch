#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP
// STAND-IN for Jeremy's  ReLU and Agatha'sSigmoid/Tanh.
#include <algorithm>
#include <cmath>
#include "cnn_library/tensor.hpp"
namespace CNN {
    struct ReLU    { Tensor forward(Tensor x) const { for (double& v : x.data) v = std::max(0.0, v); return x; } };
    struct Sigmoid { Tensor forward(Tensor x) const { for (double& v : x.data) v = 1.0 / (1.0 + std::exp(-v)); return x; } };
    struct Tanh    { Tensor forward(Tensor x) const { for (double& v : x.data) v = std::tanh(v); return x; } };
}
#endif
