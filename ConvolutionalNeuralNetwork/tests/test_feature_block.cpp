
#include "feature_block.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace std;

int main() {
    // One image, one input channel, 4x4 pixels.
    Tensor input(1, 1, 4, 4);

    for (size_t h = 0; h < 4; ++h) {
        for (size_t w = 0; w < 4; ++w) {
            input(0, 0, h, w) = 1.0;
        }
    }

    // Two filters, one input channel, 2x2 kernels.
    Tensor filters(2, 1, 2, 2);

    // Filter 0: all positive weights.
    // Each convolution position produces 4.
    for (size_t r = 0; r < 2; ++r) {
        for (size_t s = 0; s < 2; ++s) {
            filters(0, 0, r, s) = 1.0;
        }
    }

    // Filter 1: all negative weights.
    // Each convolution position produces -4.
    for (size_t r = 0; r < 2; ++r) {
        for (size_t s = 0; s < 2; ++s) {
            filters(1, 0, r, s) = -1.0;
        }
    }

    // Create a feature block with 2x2 max pooling
    // and pooling stride 2.
    cnn::FeatureBlock block(filters, 2, 2);

    // Execute convolution -> ReLU -> max pooling.
    Tensor output = block.forward(input);

    // Convolution output: (1, 2, 3, 3).
    // After pooling:      (1, 2, 1, 1).
    assert(output.shape.size() == 4);
    assert(output.shape[0] == 1);
    assert(output.shape[1] == 2);
    assert(output.shape[2] == 1);
    assert(output.shape[3] == 1);

    // Filter 0: ReLU(4) = 4, max pool = 4.
    assert(abs(output(0, 0, 0, 0) - 4.0) < 1e-9);

    // Filter 1: ReLU(-4) = 0, max pool = 0.
    assert(abs(output(0, 1, 0, 0) - 0.0) < 1e-9);

    cout << "Feature block tests passed!" << endl;
    cout << "Output shape: (1, 2, 1, 1)" << endl;
    cout << "Feature map 1: "
         << output(0, 0, 0, 0) << endl;
    cout << "Feature map 2: "
         << output(0, 1, 0, 0) << endl;

    return 0;
}
