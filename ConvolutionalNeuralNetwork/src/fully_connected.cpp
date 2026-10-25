
#include "fully_connected.hpp"

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
            throw invalid_argument(name + " contains a non-finite value");
        }
    }
}

void validateWeights(const vector<vector<double>>& weights)
{
    if (weights.empty()) {
        throw invalid_argument("Weights must contain at least one output neuron");
    }

    if (weights[0].empty()) {
        throw invalid_argument("Each output neuron must have input weights");
    }

    const size_t expected_input_size = weights[0].size();

    for (const vector<double>& row : weights) {
        if (row.size() != expected_input_size) {
            throw invalid_argument(
                "All rows in the weight matrix must have the same size");
        }

        validateVector(row, "Weights");
    }
}

} // namespace


vector<double> fullyConnectedForward(
    const vector<vector<double>>& weights,
    const vector<double>& input,
    const vector<double>& biases)
{
    validateWeights(weights);
    validateVector(input, "Input");
    validateVector(biases, "Biases");

    const size_t output_size = weights.size();
    const size_t input_size = weights[0].size();

    if (input.size() != input_size) {
        throw invalid_argument(
            "Input size must match the number of weight columns");
    }

    if (biases.size() != output_size) {
        throw invalid_argument(
            "Bias count must match the number of output neurons");
    }

    vector<double> output(output_size, 0.0);

    // Each row of weights belongs to one output neuron.
    for (size_t i = 0; i < output_size; ++i) {
        double sum = biases[i];

        // Calculate the dot product of the neuron's weights and input.
        for (size_t j = 0; j < input_size; ++j) {
            sum += weights[i][j] * input[j];
        }

        if (!isfinite(sum)) {
            throw overflow_error(
                "Non-finite result produced during forward pass");
        }

        output[i] = sum;
    }

    return output;
}


FullyConnectedLayer::FullyConnectedLayer(
    const vector<vector<double>>& layer_weights,
    const vector<double>& layer_biases)
    : weights(layer_weights), biases(layer_biases)
{
    validateWeights(weights);
    validateVector(biases, "Biases");

    if (biases.size() != weights.size()) {
        throw invalid_argument(
            "Bias count must match the number of output neurons");
    }
}


vector<double> FullyConnectedLayer::forward(
    const vector<double>& input) const
{
    return fullyConnectedForward(weights, input, biases);
}


size_t FullyConnectedLayer::inputSize() const
{
    return weights[0].size();
}


size_t FullyConnectedLayer::outputSize() const
{
    return weights.size();
}


const vector<vector<double>>& FullyConnectedLayer::getWeights() const
{
    return weights;
}


const vector<double>& FullyConnectedLayer::getBiases() const
{
    return biases;
}

} // namespace cnn
