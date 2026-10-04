#include <iostream>
#include <cassert>
#include "tensor.hpp"
#include "filter_initialization.h"
#include <filter_initialization.h>

int main() {
    std::cout << "[RUNNING] Verifying Mugarura's Filter Initialization Engine..." << std::endl;

    // Test parameters: 2 Filters, 3 Channels (RGB), 3x3 Spatial Footprint
    size_t K = 2, C = 3, Kh = 3, Kw = 3;
    
    // Execute module methods
    Tensor weights = FilterInitializer::initialize_filters(K, C, Kh, Kw, -0.2, 0.2);
    Tensor biases  = FilterInitializer::initialize_biases(K);

    // Test 1: Total Allocated Size Check
    size_t expected_elements = K * C * Kh * Kw;
    assert(weights.size() == expected_elements);
    std::cout << " -> Pass: Weights memory map matches exactly (" 
              << weights.size() << " elements)." << std::endl;

    // Test 2: 
    // Reference parameters: filter index 1, channel index 2, row index 1, column index 2
    size_t k = 1, c = 2, h = 1, w = 2;
    size_t expected_flat_index = k * (C * Kh * Kw) + c * (Kh * Kw) + h * Kw + w;
    
    
    assert(&(weights(k, c, h, w)) == &(weights.data[expected_flat_index]));
    std::cout << " -> Pass: Safely interfaces with flat indexing model." << std::endl;

    // Test 3: Structural Validation for Downstream Broadcasting 
    assert(biases.shape[0] == 1 && biases.shape[1] == K && biases.shape[2] == 1 && biases.shape[3] == 1);
    std::cout << " -> Pass: Bias structure verified at {1, " << K << ", 1, 1}." << std::endl;

    std::cout << "\n[SUCCESS] Mugarura (M5) module integrates flawlessly with upstream architecture!" << std::endl;
    return 0;
}
