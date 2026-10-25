
#include "multi_channel_convolution.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace std;

int main() {
    // Create one image with two 3x3 input channels.
    Tensor input(1, 2, 3, 3);

    // Channel 0 contains ones.
    // Channel 1 contains twos.
    for (size_t h = 0; h < 3; ++h) {
        for (size_t w = 0; w < 3; ++w) {
            input(0, 0, h, w) = 1.0;
            input(0, 1, h, w) = 2.0;
        }
    }

    // Create two filters with two channels each.
    // Each filter has a 2x2 kernel.
    Tensor filters(2, 2, 2, 2);

    // Filter 0: all weights equal 1.
    for (size_t c = 0; c < 2; ++c) {
        for (size_t r = 0; r < 2; ++r) {
            for (size_t s = 0; s < 2; ++s) {
                filters(0, c, r, s) = 1.0;
            }
        }
    }

    // Filter 1: first channel weights are 2,
    // second channel weights are 0.
    for (size_t r = 0; r < 2; ++r) {
        for (size_t s = 0; s < 2; ++s) {
            filters(1, 0, r, s) = 2.0;
            filters(1, 1, r, s) = 0.0;
        }
    }

    // Perform multi-channel convolution.
    Tensor output =
        cnn::multiChannelConvolution(input, filters);

    // Check the output shape: (1, 2, 2, 2).
    assert(output.shape.size() == 4);
    assert(output.shape[0] == 1);
    assert(output.shape[1] == 2);
    assert(output.shape[2] == 2);
    assert(output.shape[3] == 2);

    // Filter 0 result: (4 * 1) + (4 * 2) = 12.
    // Filter 1 result: (4 * 2) + (4 * 0) = 8.
    for (size_t h = 0; h < 2; ++h) {
        for (size_t w = 0; w < 2; ++w) {
            assert(
                abs(output(0, 0, h, w) - 12.0) < 1e-9
            );

            assert(
                abs(output(0, 1, h, w) - 8.0) < 1e-9
            );
        }
    }

    cout << "Multi-channel convolution tests passed!"
         << endl;

    cout << "Output shape: (1, 2, 2, 2)" << endl;

    cout << "Feature map 1 value: "
         << output(0, 0, 0, 0) << endl;

    cout << "Feature map 2 value: "
         << output(0, 1, 0, 0) << endl;

    return 0;
}
