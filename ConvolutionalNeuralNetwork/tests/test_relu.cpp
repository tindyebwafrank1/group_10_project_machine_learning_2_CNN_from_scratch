
#include <cassert>
#include <cmath>
#include <iostream>

#include "tensor.hpp"
#include "relu.hpp"

int main()
{
    std::cout << "[RUNNING] ReLU activation tests..." << std::endl;

    // Create a tensor with shape (1, 1, 2, 3).
    Tensor input(1, 1, 2, 3);

    // Fill the tensor with test values.
    input(0, 0, 0, 0) = -5.0;
    input(0, 0, 0, 1) = -2.0;
    input(0, 0, 0, 2) =  0.0;
    input(0, 0, 1, 0) =  1.0;
    input(0, 0, 1, 1) =  3.5;
    input(0, 0, 1, 2) =  7.0;

    // Test ReLU returning a new tensor.
    Tensor output = cnn::relu(input);

    assert(output.shape == input.shape);
    assert(output.size() == input.size());

    assert(output(0, 0, 0, 0) == 0.0);
    assert(output(0, 0, 0, 1) == 0.0);
    assert(output(0, 0, 0, 2) == 0.0);
    assert(output(0, 0, 1, 0) == 1.0);
    assert(output(0, 0, 1, 1) == 3.5);
    assert(output(0, 0, 1, 2) == 7.0);

    // Confirm the original input was not modified.
    assert(input(0, 0, 0, 0) == -5.0);
    assert(input(0, 0, 1, 1) == 3.5);

    // Test ReLU modifying the original tensor.
    cnn::reluInPlace(input);

    assert(input(0, 0, 0, 0) == 0.0);
    assert(input(0, 0, 0, 1) == 0.0);
    assert(input(0, 0, 1, 1) == 3.5);

    std::cout << "[PASS] All ReLU tests passed!" << std::endl;

    return 0;
}
