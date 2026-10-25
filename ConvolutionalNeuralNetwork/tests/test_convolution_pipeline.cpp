
#include <iostream>
#include <stdexcept>
#include "tensor.hpp"
#include "convolution.hpp"

using namespace std;

int main()
{
    int failures = 0;

    // Test 1: Output dimension calculation.
    if (cnn::conv_output_dim(4, 2) != 3)
    {
        cerr << "FAILED: Output dimension calculation\n";
        failures++;
    }

    // Test 2: Correct numerical convolution.
    Tensor input(1, 1, 4, 4);
    double values[4][4] = {
        {1,  2,  3,  4},
        {5,  6,  7,  8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    for (size_t h = 0; h < 4; h++)
    {
        for (size_t w = 0; w < 4; w++)
        {
            input(0, 0, h, w) = values[h][w];
        }
    }

    Tensor kernel(1, 1, 2, 2);
    kernel(0, 0, 0, 0) = 1;
    kernel(0, 0, 0, 1) = 0;
    kernel(0, 0, 1, 0) = 0;
    kernel(0, 0, 1, 1) = 1;

    Tensor output = cnn::conv2d_single(input, kernel);

    if (output.shape[2] != 3 || output.shape[3] != 3)
    {
        cerr << "FAILED: Output tensor shape\n";
        failures++;
    }

    if (output(0, 0, 0, 0) != 7 ||
        output(0, 0, 1, 1) != 17 ||
        output(0, 0, 2, 2) != 27)
    {
        cerr << "FAILED: Numerical convolution results\n";
        failures++;
    }

    // Test 3: Reject a kernel larger than the input.
    bool exception_caught = false;

    try
    {
        Tensor small_input(1, 1, 2, 2);
        Tensor large_kernel(1, 1, 3, 3);

        cnn::conv2d_single(small_input, large_kernel);
    }
    catch (const invalid_argument&)
    {
        exception_caught = true;
    }

    if (!exception_caught)
    {
        cerr << "FAILED: Oversized kernel validation\n";
        failures++;
    }

    // Test 4: Reject an invalid zero-sized kernel dimension.
    exception_caught = false;

    try
    {
        cnn::conv_output_dim(4, 0);
    }
    catch (const invalid_argument&)
    {
        exception_caught = true;
    }

    if (!exception_caught)
    {
        cerr << "FAILED: Zero-sized kernel validation\n";
        failures++;
    }

    if (failures == 0)
    {
        cout << "All convolution pipeline tests passed!\n";
        return 0;
    }

    cerr << failures << " test(s) failed.\n";
    return 1;
}
