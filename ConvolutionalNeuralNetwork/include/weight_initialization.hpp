#ifndef CNN_WEIGHT_INITIALIZATION_HPP
#define CNN_WEIGHT_INITIALIZATION_HPP

#include <cstddef>
#include <vector>

using namespace std;

namespace cnn {

enum class InitializationMethod {
    Xavier,
    He
};

// Xavier uniform or He normal weight initialization.
vector<double> initializeWeights(
    size_t fan_in,
    size_t fan_out,
    InitializationMethod method
);

// Initialize all biases to zero.
vector<double> initializeBiases(
    size_t count
);

} // namespace cnn

#endif