
#ifndef CNN_PREDICTION_HPP
#define CNN_PREDICTION_HPP

#include <cstddef>
#include <vector>

namespace cnn {

// Return the index of the class with the highest score.
// If multiple classes share the highest score, return the first index.
std::size_t argmax(const std::vector<double>& scores);

// Predict the class from a vector of output scores.
std::size_t predict(const std::vector<double>& scores);

} // namespace cnn

#endif // CNN_PREDICTION_HPP
