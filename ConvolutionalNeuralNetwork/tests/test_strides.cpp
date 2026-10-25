
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "strides.hpp"

using namespace std;
using namespace cnn;

void testCalculateStridedOutputSize()
{
    // floor((5 - 3) / 2) + 1 = 2
    assert(calculateStridedOutputSize(5, 3, 2) == 2);

    // floor((5 + 2*1 - 3) / 2) + 1 = 3
    assert(calculateStridedOutputSize(5, 3, 2, 1) == 3);

    // Kernel covers the entire input.
    assert(calculateStridedOutputSize(4, 4, 1) == 1);
}

void testSubsampleWithStrideTwo()
{
    Tensor input(1, 1, 4, 4);

    // Fill the tensor with values 1 through 16.
    double value = 1.0;

    for (size_t h = 0; h < 4; ++h) {
        for (size_t w = 0; w < 4; ++w) {
            input(0, 0, h, w) = value++;
        }
    }

    Tensor output = subsample(input, 2);

    assert(output.shape[0] == 1);
    assert(output.shape[1] == 1);
    assert(output.shape[2] == 2);
    assert(output.shape[3] == 2);

    // Samples positions (0,0), (0,2), (2,0), (2,2).
    assert(output(0, 0, 0, 0) == 1.0);
    assert(output(0, 0, 0, 1) == 3.0);
    assert(output(0, 0, 1, 0) == 9.0);
    assert(output(0, 0, 1, 1) == 11.0);
}

void testDifferentStrides()
{
    Tensor input(1, 1, 4, 5);

    for (size_t h = 0; h < 4; ++h) {
        for (size_t w = 0; w < 5; ++w) {
            input(0, 0, h, w) = h * 10 + w;
        }
    }

    Tensor output = subsample(input, 2, 3);

    assert(output.shape[2] == 2);
    assert(output.shape[3] == 2);

    assert(output(0, 0, 0, 0) == 0.0);
    assert(output(0, 0, 0, 1) == 3.0);
    assert(output(0, 0, 1, 0) == 20.0);
    assert(output(0, 0, 1, 1) == 23.0);
}

void testMultipleChannels()
{
    Tensor input(1, 2, 2, 2);

    input(0, 0, 0, 0) = 1.0;
    input(0, 0, 0, 1) = 2.0;
    input(0, 0, 1, 0) = 3.0;
    input(0, 0, 1, 1) = 4.0;

    input(0, 1, 0, 0) = 5.0;
    input(0, 1, 0, 1) = 6.0;
    input(0, 1, 1, 0) = 7.0;
    input(0, 1, 1, 1) = 8.0;

    Tensor output = subsample(input, 2);

    assert(output.shape[1] == 2);
    assert(output(0, 0, 0, 0) == 1.0);
    assert(output(0, 1, 0, 0) == 5.0);
}

void testInvalidStride()
{
    Tensor input(1, 1, 2, 2);

    bool exception_thrown = false;

    try {
        subsample(input, 0);
    } catch (const invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);
}

int main()
{
    testCalculateStridedOutputSize();
    testSubsampleWithStrideTwo();
    testDifferentStrides();
    testMultipleChannels();
    testInvalidStride();

    cout << "test_strides: all passed" << endl;

    return 0;
}
