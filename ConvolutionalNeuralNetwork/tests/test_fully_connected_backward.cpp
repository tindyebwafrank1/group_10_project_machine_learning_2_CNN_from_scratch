
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "fully_connected_backward.hpp"

using namespace std;
using namespace cnn;

bool approximatelyEqual(
    double actual,
    double expected,
    double tolerance = 1e-9)
{
    return fabs(actual - expected) < tolerance;
}

void testWeightGradients()
{
    vector<vector<double>> weights = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    vector<double> input = {5.0, 6.0};
    vector<double> delta = {0.1, 0.2};

    FullyConnectedGradients gradients =
        fullyConnectedBackward(weights, input, delta);

    // dL/dW = delta * input transpose
    //
    // [[0.1*5, 0.1*6],
    //  [0.2*5, 0.2*6]]
    assert(approximatelyEqual(gradients.d_weights[0][0], 0.5));
    assert(approximatelyEqual(gradients.d_weights[0][1], 0.6));
    assert(approximatelyEqual(gradients.d_weights[1][0], 1.0));
    assert(approximatelyEqual(gradients.d_weights[1][1], 1.2));
}

void testBiasGradients()
{
    vector<vector<double>> weights = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    vector<double> input = {5.0, 6.0};
    vector<double> delta = {0.1, 0.2};

    FullyConnectedGradients gradients =
        fullyConnectedBackward(weights, input, delta);

    // dL/db = delta
    assert(approximatelyEqual(gradients.d_biases[0], 0.1));
    assert(approximatelyEqual(gradients.d_biases[1], 0.2));
}

void testInputGradients()
{
    vector<vector<double>> weights = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    vector<double> input = {5.0, 6.0};
    vector<double> delta = {0.1, 0.2};

    FullyConnectedGradients gradients =
        fullyConnectedBackward(weights, input, delta);

    // dL/dx = transpose(W) * delta
    //
    // dx[0] = 1*0.1 + 3*0.2 = 0.7
    // dx[1] = 2*0.1 + 4*0.2 = 1.0
    assert(approximatelyEqual(gradients.d_input[0], 0.7));
    assert(approximatelyEqual(gradients.d_input[1], 1.0));
}

void testGradientDimensions()
{
    vector<vector<double>> weights = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    vector<double> input = {1.0, 2.0, 3.0};
    vector<double> delta = {0.5, -0.5};

    FullyConnectedGradients gradients =
        fullyConnectedBackward(weights, input, delta);

    assert(gradients.d_weights.size() == 2);
    assert(gradients.d_weights[0].size() == 3);
    assert(gradients.d_biases.size() == 2);
    assert(gradients.d_input.size() == 3);
}

void testInputSizeMismatch()
{
    vector<vector<double>> weights = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    vector<double> input = {1.0};
    vector<double> delta = {0.1, 0.2};

    bool exception_thrown = false;

    try {
        fullyConnectedBackward(weights, input, delta);
    } catch (const invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);
}

void testUpstreamGradientSizeMismatch()
{
    vector<vector<double>> weights = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    vector<double> input = {1.0, 2.0};
    vector<double> delta = {0.1};

    bool exception_thrown = false;

    try {
        fullyConnectedBackward(weights, input, delta);
    } catch (const invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);
}

void testInvalidWeightMatrix()
{
    vector<vector<double>> weights = {
        {1.0, 2.0},
        {3.0}
    };

    vector<double> input = {1.0, 2.0};
    vector<double> delta = {0.1, 0.2};

    bool exception_thrown = false;

    try {
        fullyConnectedBackward(weights, input, delta);
    } catch (const invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);
}

int main()
{
    testWeightGradients();
    testBiasGradients();
    testInputGradients();
    testGradientDimensions();
    testInputSizeMismatch();
    testUpstreamGradientSizeMismatch();
    testInvalidWeightMatrix();

    cout << "test_fully_connected_backward: all passed" << endl;

    return 0;
}
