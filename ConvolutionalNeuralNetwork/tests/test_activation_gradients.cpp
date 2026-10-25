
#include "activations.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace std;

bool close(double a, double b, double tolerance = 1e-9)
{
    return fabs(a - b) < tolerance;
}

int main()
{
    // Test 1: ReLU gradients
    
    act::ReLU relu;

    Tensor relu_input(1, 1, 1, 3);
    relu_input.data = {-2.0, 0.0, 3.0};

    Tensor relu_output = relu.forward(relu_input);

    Tensor relu_upstream(1, 1, 1, 3);
    relu_upstream.data = {1.0, 2.0, 4.0};

    Tensor relu_gradient =
        relu.backward(relu_output, relu_upstream);

    assert(close(relu_gradient.data[0], 0.0));
    assert(close(relu_gradient.data[1], 0.0));
    assert(close(relu_gradient.data[2], 4.0));

    // Test 2: Sigmoid gradients
    
    act::Sigmoid sigmoid;

    Tensor sigmoid_input(1, 1, 1, 3);
    sigmoid_input.data = {0.0, 1.0, -1.0};

    Tensor sigmoid_output = sigmoid.forward(sigmoid_input);

    Tensor sigmoid_upstream(1, 1, 1, 3);
    sigmoid_upstream.data = {2.0, 1.0, 3.0};

    Tensor sigmoid_gradient =
        sigmoid.backward(sigmoid_output, sigmoid_upstream);

    // At x = 0, sigmoid(x) = 0.5.
    // Incoming gradient 2 * 0.5 * (1 - 0.5) = 0.5.
    assert(close(sigmoid_gradient.data[0], 0.5));

    for (double value : sigmoid_gradient.data)
    {
        assert(isfinite(value));
    }

    // Test 3: Tanh gradients
    
    act::Tanh tanh_activation;

    Tensor tanh_input(1, 1, 1, 3);
    tanh_input.data = {0.0, 1.0, -1.0};

    Tensor tanh_output = tanh_activation.forward(tanh_input);

    Tensor tanh_upstream(1, 1, 1, 3);
    tanh_upstream.data = {3.0, 2.0, 4.0};

    Tensor tanh_gradient =
        tanh_activation.backward(tanh_output, tanh_upstream);

    // At x = 0, tanh(x) = 0.
    // Incoming gradient 3 * (1 - 0^2) = 3.
    assert(close(tanh_gradient.data[0], 3.0));

    for (double value : tanh_gradient.data)
    {
        assert(isfinite(value));
    }

   
    // Test 4: Shape mismatch must be rejected
    
    bool thrown = false;

    try
    {
        Tensor wrong_shape(1, 1, 1, 2);
        relu.backward(relu_output, wrong_shape);
    }
    catch (const invalid_argument&)
    {
        thrown = true;
    }

    assert(thrown);

    
    // Test 5: Forward operations do not modify input
    
    assert(relu_input.data[0] == -2.0);
    assert(sigmoid_input.data[0] == 0.0);
    assert(tanh_input.data[0] == 0.0);

    cout << "All activation gradient tests passed!" << endl;

    return 0;
}
