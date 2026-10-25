
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "fully_connected.hpp"

using namespace std;
using namespace cnn;

bool approximatelyEqual(double a, double b, double tolerance = 1e-9)
{
    return fabs(a - b) < tolerance;
}

void testBasicForwardPass()
{
    // Two input features and two output neurons.
    vector<vector<double>> weights = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    vector<double> input = {5.0, 6.0};
    vector<double> biases = {1.0, 2.0};

    vector<double> output =
        fullyConnectedForward(weights, input, biases);

    // z1 = (1*5) + (2*6) + 1 = 18
    // z2 = (3*5) + (4*6) + 2 = 41
    assert(output.size() == 2);
    assert(approximatelyEqual(output[0], 18.0));
    assert(approximatelyEqual(output[1], 41.0));
}

void testZeroBiases()
{
    vector<vector<double>> weights = {
        {2.0, 3.0},
        {4.0, 5.0}
    };

    vector<double> input = {1.0, 2.0};
    vector<double> biases = {0.0, 0.0};

    vector<double> output =
        fullyConnectedForward(weights, input, biases);

    assert(approximatelyEqual(output[0], 8.0));
    assert(approximatelyEqual(output[1], 14.0));
}

void testSingleOutputNeuron()
{
    vector<vector<double>> weights = {
        {2.0, -1.0, 3.0}
    };

    vector<double> input = {4.0, 5.0, 2.0};
    vector<double> biases = {1.0};

    vector<double> output =
        fullyConnectedForward(weights, input, biases);

    // z = (2*4) + (-1*5) + (3*2) + 1 = 10
    assert(output.size() == 1);
    assert(approximatelyEqual(output[0], 10.0));
}

void testLayerClass()
{
    vector<vector<double>> weights = {
        {1.0, 0.0},
        {0.0, 1.0}
    };

    vector<double> biases = {0.5, -0.5};

    FullyConnectedLayer layer(weights, biases);

    vector<double> output = layer.forward({3.0, 7.0});

    assert(layer.inputSize() == 2);
    assert(layer.outputSize() == 2);

    assert(approximatelyEqual(output[0], 3.5));
    assert(approximatelyEqual(output[1], 6.5));
}

void testInputSizeMismatch()
{
    vector<vector<double>> weights = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    vector<double> input = {5.0};
    vector<double> biases = {0.0, 0.0};

    bool exception_thrown = false;

    try {
        fullyConnectedForward(weights, input, biases);
    } catch (const invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);
}

void testBiasSizeMismatch()
{
    vector<vector<double>> weights = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    vector<double> input = {1.0, 2.0};
    vector<double> biases = {0.0};

    bool exception_thrown = false;

    try {
        fullyConnectedForward(weights, input, biases);
    } catch (const invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);
}

void testInconsistentWeightRows()
{
    vector<vector<double>> weights = {
        {1.0, 2.0},
        {3.0}
    };

    vector<double> input = {1.0, 2.0};
    vector<double> biases = {0.0, 0.0};

    bool exception_thrown = false;

    try {
        fullyConnectedForward(weights, input, biases);
    } catch (const invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);
}

int main()
{
    testBasicForwardPass();
    testZeroBiases();
    testSingleOutputNeuron();
    testLayerClass();
    testInputSizeMismatch();
    testBiasSizeMismatch();
    testInconsistentWeightRows();

    cout << "test_fully_connected: all passed" << endl;

    return 0;
}
