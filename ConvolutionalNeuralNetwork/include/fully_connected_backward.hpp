
#ifndef CNN_FULLY_CONNECTED_BACKWARD_HPP
#define CNN_FULLY_CONNECTED_BACKWARD_HPP

#include <vector>

using namespace std;

namespace cnn {

// Stores the gradients calculated during backpropagation.
struct FullyConnectedGradients {
    vector<vector<double>> d_weights;
    vector<double> d_biases;
    vector<double> d_input;
};

// Calculates gradients for a single example.
//
// weights:           [output_size][input_size]
// input:             [input_size]
// upstream_gradient: dL/dz, with [output_size] elements
//
// Returns:
// d_weights = dL/dW
// d_biases  = dL/db
// d_input   = dL/dx
FullyConnectedGradients fullyConnectedBackward(
    const vector<vector<double>>& weights,
    const vector<double>& input,
    const vector<double>& upstream_gradient
);

} // namespace cnn

#endif
