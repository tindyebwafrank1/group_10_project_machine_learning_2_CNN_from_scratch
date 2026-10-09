#include <iostream>
#include <cassert>
#include <cmath>
#include "tensor.hpp"
#include "max_pooling.hpp"

int main() {
    std::cout << "[RUNNING] Verifying Mugarura's Max Pooling Layer with Index Tracking..." << std::endl;

    size_t N = 1, C = 1, H = 4, W = 4;
    Tensor input(N, C, H, W);

    input(0, 0, 0, 0) = 1.0; input(0, 0, 0, 1) = 3.0;
    input(0, 0, 1, 0) = 2.0; input(0, 0, 1, 1) = 9.0;

    input(0, 0, 0, 2) = 4.0; input(0, 0, 0, 3) = 12.0;
    input(0, 0, 1, 2) = 0.0; input(0, 0, 1, 3) = -1.5;

    input(0, 0, 2, 0) = 7.5; input(0, 0, 2, 1) = 4.0;
    input(0, 0, 3, 0) = 5.0; input(0, 0, 3, 1) = 6.0;

    input(0, 0, 2, 2) = 8.0; input(0, 0, 2, 3) = 11.0;
    input(0, 0, 3, 2) = 2.0; input(0, 0, 3, 3) = 15.0;

    size_t pool_size = 2;
    size_t stride = 2;
    MaxPooling pool_layer(pool_size, pool_size, stride);

    Tensor output = pool_layer.forward(input);

    assert(output.shape[0] == 1);
    assert(output.shape[1] == 1);
    assert(output.shape[2] == 2);
    assert(output.shape[3] == 2);
    std::cout << " -> Pass: Output spatial configuration is perfectly downsampled to 2x2." << std::endl;

    assert(std::abs(output(0, 0, 0, 0) - 9.0) < 1e-6);
    assert(std::abs(output(0, 0, 0, 1) - 12.0) < 1e-6);
    assert(std::abs(output(0, 0, 1, 0) - 7.5) < 1e-6);
    assert(std::abs(output(0, 0, 1, 1) - 15.0) < 1e-6);
    std::cout << " -> Pass: Matrix mathematical pooling calculations match local patch values." << std::endl;

    // Row-major input indices for the maxima: 5, 3, 8, and 15.
    assert(pool_layer.max_indices[0] == 5);
    assert(pool_layer.max_indices[1] == 3);
    assert(pool_layer.max_indices[2] == 8);
    assert(pool_layer.max_indices[3] == 15);
    std::cout << " -> Pass: Argmax indices correctly recorded for future gradient back-routing." << std::endl;

    std::cout << "\n[SUCCESS] Mugarura (M5) Max Pooling module meets all Week 2 validation metrics!" << std::endl;
    return 0;
}