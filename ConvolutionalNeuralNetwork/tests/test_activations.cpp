
#include "activations.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;

bool approximatelyEqual(double a, double b)
{
    return abs(a - b) < 1e-9;
}

int main()
{
    int failures = 0;

    // Prepare input tensor.
    Tensor input(1, 1, 1, 3);
    input(0, 0, 0, 0) = -2.0;
    input(0, 0, 0, 1) = 0.0;
    input(0, 0, 0, 2) = 2.0;

    // Test 1: Sigmoid forward.
    act::Sigmoid sigmoid;
    Tensor sig = sigmoid.forward(input);

    if (!approximatelyEqual(sig.data[0], 1.0 / (1.0 + exp(2.0))) ||
        !approximatelyEqual(sig.data[1], 0.5) ||
        !approximatelyEqual(sig.data[2], 1.0 / (1.0 + exp(-2.0))))
    {
        cerr << "FAILED: Sigmoid forward\n";
        failures++;
    }

    // Test 2: Tanh forward.
    act::Tanh tanhLayer;
    Tensor tanhOutput = tanhLayer.forward(input);

    if (!approximatelyEqual(tanhOutput.data[0], std::tanh(-2.0)) ||
        !approximatelyEqual(tanhOutput.data[1], 0.0) ||
        !approximatelyEqual(tanhOutput.data[2], std::tanh(2.0)))
    {
        cerr << "FAILED: Tanh forward\n";
        failures++;
    }

    // Test 3: ReLU forward.
    act::ReLU relu;
    Tensor reluOutput = relu.forward(input);

    if (!approximatelyEqual(reluOutput.data[0], 0.0) ||
        !approximatelyEqual(reluOutput.data[1], 0.0) ||
        !approximatelyEqual(reluOutput.data[2], 2.0))
    {
        cerr << "FAILED: ReLU forward\n";
        failures++;
    }

    // Test 4: Input must remain unchanged.
    if (!approximatelyEqual(input.data[0], -2.0) ||
        !approximatelyEqual(input.data[1], 0.0) ||
        !approximatelyEqual(input.data[2], 2.0))
    {
        cerr << "FAILED: Input was modified\n";
        failures++;
    }

    // Upstream gradients for backward tests.
    Tensor gradOut(1, 1, 1, 3);
    gradOut.fill(1.0);

    // Test 5: Sigmoid backward.
    Tensor sigGrad = sigmoid.backward(sig, gradOut);

    for (size_t i = 0; i < sig.size(); ++i)
    {
        double y = sig.data[i];
        double expected = y * (1.0 - y);

        if (!approximatelyEqual(sigGrad.data[i], expected))
        {
            cerr << "FAILED: Sigmoid backward\n";
            failures++;
            break;
        }
    }

    // Test 6: Tanh backward.
    Tensor tanhGrad = tanhLayer.backward(tanhOutput, gradOut);

    for (size_t i = 0; i < tanhOutput.size(); ++i)
    {
        double y = tanhOutput.data[i];
        double expected = 1.0 - y * y;

        if (!approximatelyEqual(tanhGrad.data[i], expected))
        {
            cerr << "FAILED: Tanh backward\n";
            failures++;
            break;
        }
    }

    // Test 7: ReLU backward.
    Tensor reluGrad = relu.backward(reluOutput, gradOut);

    if (!approximatelyEqual(reluGrad.data[0], 0.0) ||
        !approximatelyEqual(reluGrad.data[1], 0.0) ||
        !approximatelyEqual(reluGrad.data[2], 1.0))
    {
        cerr << "FAILED: ReLU backward\n";
        failures++;
    }

    // Test 8: Extreme values should remain finite.
    Tensor extreme(1, 1, 1, 2);
    extreme.data[0] = -1000.0;
    extreme.data[1] = 1000.0;

    Tensor extremeSig = sigmoid.forward(extreme);
    Tensor extremeTanh = tanhLayer.forward(extreme);

    if (!isfinite(extremeSig.data[0]) ||
        !isfinite(extremeSig.data[1]) ||
        !isfinite(extremeTanh.data[0]) ||
        !isfinite(extremeTanh.data[1]) ||
        !approximatelyEqual(extremeSig.data[0], 0.0) ||
        !approximatelyEqual(extremeSig.data[1], 1.0) ||
        !approximatelyEqual(extremeTanh.data[0], -1.0) ||
        !approximatelyEqual(extremeTanh.data[1], 1.0))
    {
        cerr << "FAILED: Extreme input stability\n";
        failures++;
    }

    // Test 9: Backward must reject mismatched shapes.
    bool threw = false;

    try
    {
        Tensor wrongShape(1, 1, 3, 1);
        sigmoid.backward(sig, wrongShape);
    }
    catch (const invalid_argument&)
    {
        threw = true;
    }

    if (!threw)
    {
        cerr << "FAILED: Shape mismatch was not rejected\n";
        failures++;
    }

    if (failures == 0)
    {
        cout << "All activation tests passed!" << endl;
        cout << "Sigmoid(0) = " << sig.data[1] << endl;
        cout << "Tanh(0) = " << tanhOutput.data[1] << endl;
        return 0;
    }

    cerr << failures << " test(s) failed." << endl;
    return 1;
}
