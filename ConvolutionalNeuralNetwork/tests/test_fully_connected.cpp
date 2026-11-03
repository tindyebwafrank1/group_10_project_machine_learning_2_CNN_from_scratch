
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <limits>

#include "fully_connected.hpp"
#include "tensor.hpp"

using namespace std;

bool approximatelyEqual(double a, double b, double tolerance = 1e-9) {
    return fabs(a - b) < tolerance;
}

// Test 1: Basic forward pass
void testBasicForwardPass() {
    cnn::FullyConnected layer(2, 2);

    // Output neuron 1: weights {1, 2}
    // Output neuron 2: weights {3, 4}
    layer.setWeights({1.0, 2.0, 3.0, 4.0});
    layer.setBias({1.0, 2.0});

    Tensor input(1, 2, 1, 1);
    input.data = {5.0, 6.0};

    Tensor output = layer.forward(input);

    assert(output.shape == vector<size_t>({1, 2, 1, 1}));
    assert(approximatelyEqual(output.data[0], 18.0));
    assert(approximatelyEqual(output.data[1], 41.0));
}

// Test 2: Zero biases
void testZeroBiases() {
    cnn::FullyConnected layer(2, 2);

    layer.setWeights({2.0, 3.0, 4.0, 5.0});
    layer.setBias({0.0, 0.0});

    Tensor input(1, 2, 1, 1);
    input.data = {1.0, 2.0};

    Tensor output = layer.forward(input);

    assert(approximatelyEqual(output.data[0], 8.0));
    assert(approximatelyEqual(output.data[1], 14.0));
}

// Test 3: Single output neuron
void testSingleOutputNeuron() {
    cnn::FullyConnected layer(3, 1);

    layer.setWeights({2.0, -1.0, 3.0});
    layer.setBias({1.0});

    Tensor input(1, 3, 1, 1);
    input.data = {4.0, 5.0, 2.0};

    Tensor output = layer.forward(input);

    assert(output.data.size() == 1);
    assert(approximatelyEqual(output.data[0], 10.0));
}

// Test 4: Layer accessors and parameter count
void testLayerAccessors() {
    cnn::FullyConnected layer(2, 3);

    assert(layer.inFeatures() == 2);
    assert(layer.outFeatures() == 3);
    assert(layer.parameterCount() == 9);
}

// Test 5: Inference
void testInference() {
    cnn::FullyConnected layer(2, 2);

    layer.setWeights({1.0, 0.0, 0.0, 1.0});
    layer.setBias({0.5, -0.5});

    Tensor input(1, 2, 1, 1);
    input.data = {3.0, 7.0};

    Tensor output = layer.infer(input);

    assert(approximatelyEqual(output.data[0], 3.5));
    assert(approximatelyEqual(output.data[1], 6.5));
    assert(!layer.hasCachedInput());
}

// Test 6: Batch processing
void testBatchProcessing() {
    cnn::FullyConnected layer(2, 1);

    layer.setWeights({2.0, 3.0});
    layer.setBias({1.0});

    Tensor input(2, 2, 1, 1);
    input.data = {1.0, 2.0, 3.0, 4.0};

    Tensor output = layer.forward(input);

    assert(output.shape[0] == 2);
    assert(output.shape[1] == 1);

    assert(approximatelyEqual(output.data[0], 9.0));
    assert(approximatelyEqual(output.data[1], 19.0));
}

// Test 7: Input feature mismatch
void testInputSizeMismatch() {
    cnn::FullyConnected layer(3, 2);

    Tensor input(1, 2, 1, 1);

    bool exceptionThrown = false;

    try {
        layer.forward(input);
    }
    catch (const invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test 8: Invalid weight count
void testWeightSizeMismatch() {
    cnn::FullyConnected layer(2, 2);

    bool exceptionThrown = false;

    try {
        layer.setWeights({1.0, 2.0});
    }
    catch (const invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test 9: Invalid bias count
void testBiasSizeMismatch() {
    cnn::FullyConnected layer(2, 2);

    bool exceptionThrown = false;

    try {
        layer.setBias({1.0});
    }
    catch (const invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test 10: Non-finite input
void testNonFiniteInput() {
    cnn::FullyConnected layer(2, 1);

    Tensor input(1, 2, 1, 1);
    input.data = {
        1.0,
        numeric_limits<double>::quiet_NaN()
    };

    bool exceptionThrown = false;

    try {
        layer.forward(input);
    }
    catch (const invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test 11: Cache management
void testCacheManagement() {
    cnn::FullyConnected layer(2, 1);

    Tensor input(1, 2, 1, 1);
    input.data = {2.0, 3.0};

    layer.forward(input);

    assert(layer.hasCachedInput());
    assert(layer.cachedInput().data == input.data);

    layer.clearCache();

    assert(!layer.hasCachedInput());

    bool exceptionThrown = false;

    try {
        layer.cachedInput();
    }
    catch (const logic_error&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

// Test 12: Invalid layer dimensions
void testInvalidDimensions() {
    bool exceptionThrown = false;

    try {
        cnn::FullyConnected layer(0, 2);
    }
    catch (const invalid_argument&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

int main() {
    testBasicForwardPass();
    testZeroBiases();
    testSingleOutputNeuron();
    testLayerAccessors();
    testInference();
    testBatchProcessing();
    testInputSizeMismatch();
    testWeightSizeMismatch();
    testBiasSizeMismatch();
    testNonFiniteInput();
    testCacheManagement();
    testInvalidDimensions();

    cout << "test_fully_connected: all 12 tests passed" << endl;

    return 0;
}
