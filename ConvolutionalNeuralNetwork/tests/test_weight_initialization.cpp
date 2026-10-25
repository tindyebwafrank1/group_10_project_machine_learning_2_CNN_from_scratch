#include "weight_initialization.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace std;

int main() {
    const size_t fan_in = 4;
    const size_t fan_out = 3;

    // Test Xavier initialization.
    const auto xavier = cnn::initializeWeights(
        fan_in,
        fan_out,
        cnn::InitializationMethod::Xavier
    );

    assert(xavier.size() == fan_in * fan_out);

    const double xavier_limit = sqrt(
        6.0 / static_cast<double>(fan_in + fan_out)
    );

    for (double weight : xavier) {
        assert(isfinite(weight));
        assert(weight >= -xavier_limit);
        assert(weight <= xavier_limit);
    }

    // Test He initialization.
    const auto he = cnn::initializeWeights(
        fan_in,
        fan_out,
        cnn::InitializationMethod::He
    );

    assert(he.size() == fan_in * fan_out);

    for (double weight : he) {
        assert(isfinite(weight));
    }

    // Test zero-initialized biases.
    const auto biases = cnn::initializeBiases(fan_out);

    assert(biases.size() == fan_out);

    for (double bias : biases) {
        assert(bias == 0.0);
    }

    // Test invalid dimensions.
    bool exception_caught = false;

    try {
        cnn::initializeWeights(
            0, fan_out,
            cnn::InitializationMethod::Xavier
        );
    }
    catch (const invalid_argument&) {
        exception_caught = true;
    }

    assert(exception_caught);

    cout << "Xavier initialization test passed!\n";
    cout << "He initialization test passed!\n";
    cout << "Bias initialization test passed!\n";
    cout << "Invalid dimensions test passed!\n";

    return 0;
}