
#include "fully_connected_backward.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

using namespace std;

namespace cnn {

namespace {

void validateVector(const vector<double>& values, const string& name)
{
    for (double value : values) {
        if (!isfinite(value)) {
            throw invalid_argument(
                name + " contains a non-finite value");
        }
    }
}

void validateWeights(const vector<vector<double>>& weights)
{
    if (weights.empty()) {
        throw invalid_argument(
            "Weights must contain at least one output neuron");
    }

    if (weights[0].empty()) {
        throw invalid_argument(
            "Each output neuron must have at least one input weight");
    }

    const size_t expected_size = weights[0].size();

    for (const vector<double>& row : weights) {
        if (row.size() != expected_size) {
            throw invalid_argument(
                "All weight rows must have the same number of elements");
        }

        validateVector(row, "Weights");
    }
}

} // namespace


FullyConnectedGradients fullyConnectedBackward(
    const vector<vector<double>>& weights,
    const vector<double>& input,
    const vector<double>& upstream_gradient)
{
    validateWeights(weights);
    validateVector(input, "Input");
    validateVector(upstream_gradient, "Upstream gradient");

    const size_t output_size = weights.size();
    const size_t input_size = weights[0].size();

    if (input.size() != input_size) {
        throw invalid_argument(
            "Input size must match the number of weight columns");
    }

    if (upstream_gradient.size() != output_size) {
        throw invalid_argument(
            "Upstream gradient size must match the number of output neurons");
    }

    FullyConnectedGradients gradients;

    // Initialize the gradient containers.
    gradients.d_weights.assign(
        output_size, vector<double>(input_size, 0.0));

    gradients.d_biases.assign(output_size, 0.0);
    gradients.d_input.assign(input_size, 0.0);

    // dL/dW[i][j] = dL/dz[i] * x[j]
    // dL/db[i]    = dL/dz[i]
    for (size_t i = 0; i < output_size; ++i) {
        const double delta = upstream_gradient[i];

        gradients.d_biases[i] = delta;

        for (size_t j = 0; j < input_size; ++j) {
            gradients.d_weights[i][j] = delta * input[j];
        }
    }

    // dL/dx[j] = sum over i of W[i][j] * dL/dz[i]
    for (size_t j = 0; j < input_size; ++j) {
        double sum = 0.0;

        for (size_t i = 0; i < output_size; ++i) {
            sum += weights[i][j] * upstream_gradient[i];
        }

        if (!isfinite(sum)) {
            throw overflow_error(
                "Non-finite input gradient produced");
        }

        gradients.d_input[j] = sum;
    }

    // Detect overflow in the calculated weight and bias gradients.
    for (const vector<double>& row : gradients.d_weights) {
        validateVector(row, "Weight gradients");
    }

    validateVector(gradients.d_biases, "Bias gradients");

    return gradients;
}

} // namespace cnn
