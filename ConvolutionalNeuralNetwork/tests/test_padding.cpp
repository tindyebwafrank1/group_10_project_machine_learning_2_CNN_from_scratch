
#include <iostream>
#include <cassert>
#include <stdexcept>

#include "tensor.hpp"
#include "padding.hpp"

using namespace std;
using namespace cnn;

void testZeroPad()
{
    cout << "Testing zeroPad()..." << endl;

    // Create a 4D tensor: N=1, C=1, H=2, W=2.
    Tensor input(1, 1, 2, 2);

    // Assign values to the input tensor.
    input(0, 0, 0, 0) = 1.0;
    input(0, 0, 0, 1) = 2.0;
    input(0, 0, 1, 0) = 3.0;
    input(0, 0, 1, 1) = 4.0;

    // Add one row and column of padding on every side.
    Tensor output = zeroPad(input, 1, 1);

    // Check output dimensions.
    assert(output.shape[0] == 1);
    assert(output.shape[1] == 1);
    assert(output.shape[2] == 4);
    assert(output.shape[3] == 4);

    // Check that original values remain in the center.
    assert(output(0, 0, 1, 1) == 1.0);
    assert(output(0, 0, 1, 2) == 2.0);
    assert(output(0, 0, 2, 1) == 3.0);
    assert(output(0, 0, 2, 2) == 4.0);

    // Check that the padding contains zeros.
    assert(output(0, 0, 0, 0) == 0.0);
    assert(output(0, 0, 0, 2) == 0.0);
    assert(output(0, 0, 3, 3) == 0.0);
    assert(output(0, 0, 1, 0) == 0.0);

    cout << "zeroPad() passed!" << endl;
}

void testNoPadding()
{
    cout << "Testing zeroPad() with no padding..." << endl;

    Tensor input(1, 1, 2, 2);

    input(0, 0, 0, 0) = 5.0;
    input(0, 0, 0, 1) = 6.0;
    input(0, 0, 1, 0) = 7.0;
    input(0, 0, 1, 1) = 8.0;

    Tensor output = zeroPad(input, 0, 0);

    // Dimensions should remain unchanged.
    assert(output.shape[2] == 2);
    assert(output.shape[3] == 2);

    // Values should remain unchanged.
    assert(output(0, 0, 0, 0) == 5.0);
    assert(output(0, 0, 0, 1) == 6.0);
    assert(output(0, 0, 1, 0) == 7.0);
    assert(output(0, 0, 1, 1) == 8.0);

    cout << "No-padding test passed!" << endl;
}

void testSamePadding()
{
    cout << "Testing calculateSamePadding()..." << endl;

    size_t top, bottom, left, right;

    // Input 5x5, kernel 3x3, stride 1.
    calculateSamePadding(
        5, 5, 3, 3, 1, 1,
        top, bottom, left, right
    );

    assert(top == 1);
    assert(bottom == 1);
    assert(left == 1);
    assert(right == 1);

    cout << "Same-padding test 1 passed!" << endl;

    // Input 4x4, kernel 3x3, stride 2.
    calculateSamePadding(
        4, 4, 3, 3, 2, 2,
        top, bottom, left, right
    );

    assert(top == 0);
    assert(bottom == 1);
    assert(left == 0);
    assert(right == 1);

    cout << "Same-padding test 2 passed!" << endl;
}

void testInvalidStride()
{
    cout << "Testing invalid stride..." << endl;

    size_t top, bottom, left, right;
    bool exceptionThrown = false;

    try
    {
        // A stride of zero is invalid.
        calculateSamePadding(
            5, 5, 3, 3, 0, 1,
            top, bottom, left, right
        );
    }
    catch (const invalid_argument&)
    {
        exceptionThrown = true;
    }

    assert(exceptionThrown);

    cout << "Invalid-stride test passed!" << endl;
}

int main()
{
    testZeroPad();
    testNoPadding();
    testSamePadding();
    testInvalidStride();

    cout << "\nAll padding tests passed successfully!" << endl;

    return 0;
}
