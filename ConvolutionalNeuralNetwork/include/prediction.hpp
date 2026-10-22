#ifndef PREDICTION_HPP
#define PREDICTION_HPP

#include <cstddef>
#include <vector>
#include "tensor.hpp"

namespace cnn {

// Index of the largest value in scores. Ties go to the lowest index.
// Throws std::invalid_argument if scores is empty or contains NaN.
std::size_t argmax(const std::vector<double>& scores);

// Predicted class for every sample in a batch. Each sample's scores are the
// consecutive size/N values of the tensor (e.g. softmax output of shape
// (N, K, 1, 1)). Returns N class indices.
// Throws std::invalid_argument if the tensor is empty or N == 0.
std::vector<std::size_t> predict(const Tensor& scores);

}  // namespace cnn

#endif  //PREDICTION_HPP