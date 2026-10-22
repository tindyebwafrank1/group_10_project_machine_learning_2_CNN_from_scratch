#include "loss_gradient.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

int main()
{
    // Predicted probabilities for three classes.
    const vector<double> probabilities = {
        0.7, 0.2, 0.1
    };

    // Target labels: class 0 is the correct class.
    const vector<double> targets = {
        1.0, 0.0, 0.0
    };

    // Calculate the combined Softmax and cross-entropy gradient.
    const auto gradient =
        cnn::softmaxCrossEntropyGradient(
            probabilities, targets);

    // Verify the expected gradients.
    assert(abs(gradient[0] - (-0.3)) < 1e-9);
    assert(abs(gradient[1] - 0.2) < 1e-9);
    assert(abs(gradient[2] - 0.1) < 1e-9);

    // Define an upstream gradient for the Softmax layer.
    const vector<double> upstream = {
        1.0, 0.0, 0.0
    };

    // Calculate the gradient through Softmax.
    const auto softmax_gradient =
        cnn::softmaxBackward(probabilities, upstream);

    // Verify the expected Softmax gradients.
    assert(abs(softmax_gradient[0] - 0.21) < 1e-9);
    assert(abs(softmax_gradient[1] - (-0.14)) < 1e-9);
    assert(abs(softmax_gradient[2] - (-0.07)) < 1e-9);

    // Initialize a flag for invalid-input testing.
    bool threw = false;

    try {
        // Pass vectors of unequal sizes to trigger an exception.
        cnn::softmaxCrossEntropyGradient(
            {0.5, 0.5}, {1.0});
    }
    catch (const invalid_argument&) {
        // Confirm that the invalid input was rejected.
        threw = true;
    }

    // Verify that the exception was thrown.
    assert(threw);

    // Display a message if all tests pass.
    cout << endl<<"All loss gradient tests passed." <<endl<< endl;

    return 0; 
}