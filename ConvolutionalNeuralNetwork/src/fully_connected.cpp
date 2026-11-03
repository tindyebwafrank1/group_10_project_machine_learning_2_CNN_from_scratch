#include "fully_connected.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace cnn {

// Validate numerical values
static void validateValues(const vector<double>& values, const string& name)
{
    for (double value : values) {
        if (!isfinite(value)) {
            throw invalid_argument(name + " contains a non-finite value");
        }
    }
}

// Original vector-based fully connected forward operation
vector<double> fullyConnectedForward(
    const vector<vector<double>>& weights,
    const vector<double>& input,
    const vector<double>& biases)
{
    if (weights.empty() || weights[0].empty()) {
        throw invalid_argument("Weights cannot be empty");
    }

    size_t output_size = weights.size();
    size_t input_size = weights[0].size();

    if (input.size() != input_size) {
        throw invalid_argument("Input size does not match weights");
    }

    if (biases.size() != output_size) {
        throw invalid_argument("Bias size does not match weights");
    }

    validateValues(input, "Input");
    validateValues(biases, "Biases");

    vector<double> output(output_size, 0.0);

    for (size_t i = 0; i < output_size; i++) {

        if (weights[i].size() != input_size) {
            throw invalid_argument("Invalid weight matrix dimensions");
        }

        validateValues(weights[i], "Weights");

        double sum = biases[i];

        for (size_t j = 0; j < input_size; j++) {
            sum += weights[i][j] * input[j];
        }

        if (!isfinite(sum)) {
            throw overflow_error("Non-finite output during forward pass");
        }

        output[i] = sum;
    }

    return output;
}

} // namespace cnn
