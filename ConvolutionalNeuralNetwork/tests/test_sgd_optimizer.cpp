
#include "sgd_optimizer.hpp"

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <limits>

using namespace std;

bool approximatelyEqual(double a, double b) {
    return abs(a - b) < 1e-9;
}

int main() {

    // Test 1: Basic parameter update.
    cnn::SGDOptimizer optimizer(0.1);

    vector<double> weights = {2.0, 4.0, 6.0};
    vector<double> weight_gradients = {0.5, 1.0, -2.0};

    optimizer.step(weights, weight_gradients);

    if (!approximatelyEqual(weights[0], 1.95) ||
        !approximatelyEqual(weights[1], 3.9) ||
        !approximatelyEqual(weights[2], 6.2)) {

        cerr << "Basic SGD update test failed.\n";
        return 1;
    }

    // Test 2: Biases can use the same optimizer.
    vector<double> biases = {1.0, 2.0};
    vector<double> bias_gradients = {0.2, -0.4};

    optimizer.step(biases, bias_gradients);

    if (!approximatelyEqual(biases[0], 0.98) ||
        !approximatelyEqual(biases[1], 2.04)) {

        cerr << "Bias update test failed.\n";
        return 1;
    }

    // Test 3: Zero gradients should not change parameters.
    vector<double> unchanged = {3.0, 5.0};
    vector<double> zero_gradients = {0.0, 0.0};

    optimizer.step(unchanged, zero_gradients);

    if (!approximatelyEqual(unchanged[0], 3.0) ||
        !approximatelyEqual(unchanged[1], 5.0)) {

        cerr << "Zero-gradient test failed.\n";
        return 1;
    }

    // Test 4: Mismatched vector sizes must throw an exception.
    try {
        vector<double> parameters = {1.0, 2.0};
        vector<double> gradients = {0.5};

        optimizer.step(parameters, gradients);

        cerr << "Mismatched-size test failed.\n";
        return 1;

    } catch (const invalid_argument&) {
        // Expected.
    }

    // Test 5: An invalid learning rate must be rejected.
    try {
        cnn::SGDOptimizer invalid_optimizer(0.0);

        cerr << "Invalid learning-rate test failed.\n";
        return 1;

    } catch (const invalid_argument&) {
        // Expected.
    }

    // Test 6: A non-finite gradient must be rejected.
    try {
        vector<double> parameters = {1.0};
        vector<double> gradients = {
            numeric_limits<double>::infinity()
        };

        optimizer.step(parameters, gradients);

        cerr << "Non-finite-gradient test failed.\n";
        return 1;

    } catch (const invalid_argument&) {
        // Expected.
    }

    // Test 7: Learning rate can be changed.
    optimizer.setLearningRate(0.01);

    if (!approximatelyEqual(optimizer.getLearningRate(), 0.01)) {
        cerr << "Learning-rate setter test failed.\n";
        return 1;
    }

    cout << "All SGD optimizer tests passed successfully.\n";

    return 0;
}
