#include <cassert>
#include <cmath>
#include <iostream>

#include "tensor.hpp"
#include "max_pooling.hpp"

int main()
{
    std::cout << "[RUNNING] Max Pooling tests..." << std::endl;

    // Create a tensor with shape (N, C, H, W) = (1, 1, 4, 4).
    Tensor input(1, 1, 4, 4);

    // Fill the input tensor with test values.
    double values[4][4] = {
        {1.0, 2.0, 4.0, 12.0},
        {3.0, 9.0, 5.0, 6.0},
        {7.5, 1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0, 15.0}
    };

    for (std::size_t h = 0; h < 4; ++h) {
        for (std::size_t w = 0; w < 4; ++w) {
            input(0, 0, h, w) = values[h][w];
        }
    }

    // Use a 2x2 pooling window with stride 2.
    MaxPooling pooling(2, 2, 2);

    // Perform max pooling.
    Tensor output = pooling.forward(input);

    // Check the output dimensions.
    assert(output.shape.size() == 4);
    assert(output.shape[0] == 1);
    assert(output.shape[1] == 1);
    assert(output.shape[2] == 2);
    assert(output.shape[3] == 2);

    // Expected output:
    // [9.0, 12.0]
    // [7.5, 15.0]

    assert(std::abs(output(0, 0, 0, 0) - 9.0) < 1e-9);
    assert(std::abs(output(0, 0, 0, 1) - 12.0) < 1e-9);
    assert(std::abs(output(0, 0, 1, 0) - 7.5) < 1e-9);
    assert(std::abs(output(0, 0, 1, 1) - 15.0) < 1e-9);

    // Check the flattened indices of the maximum values.
    // These assume indices refer to positions in the 4x4 spatial input.
    assert(pooling.max_indices.size() == 4);

    assert(pooling.max_indices[0] == 5);
    assert(pooling.max_indices[1] == 3);
    assert(pooling.max_indices[2] == 8);
    assert(pooling.max_indices[3] == 15);

    // Confirm the input tensor remains unchanged.
    assert(std::abs(input(0, 0, 0, 1) - 2.0) < 1e-9);
    assert(std::abs(input(0, 0, 1, 1) - 9.0) < 1e-9);

    std::cout << "[PASS] Output dimensions are correct." << std::endl;
    std::cout << "[PASS] Maximum values are correct." << std::endl;
    std::cout << "[PASS] Maximum indices are correct." << std::endl;
    std::cout << "[PASS] Input tensor remains unchanged." << std::endl;
    std::cout << "[PASS] All Max Pooling tests passed!" << std::endl;

    return 0;
}