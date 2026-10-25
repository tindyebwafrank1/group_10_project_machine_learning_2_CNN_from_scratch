
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>

#include "tensor.hpp"
#include "convolution_backward.hpp"

using namespace std;

bool approximatelyEqual(double actual, double expected)
{
    return fabs(actual - expected) < 1e-9;
}

int main()
{
    try {
        // Input tensor: shape (1, 1, 3, 3).
        ::Tensor input(1, 1, 3, 3);

        double input_values[3][3] = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };

        for (size_t h = 0; h < 3; ++h) {
            for (size_t w = 0; w < 3; ++w) {
                input(0, 0, h, w) = input_values[h][w];
            }
        }

        // Kernel tensor: shape (1, 1, 2, 2).
        ::Tensor kernels(1, 1, 2, 2);

        kernels(0, 0, 0, 0) = 1;
        kernels(0, 0, 0, 1) = 2;
        kernels(0, 0, 1, 0) = 3;
        kernels(0, 0, 1, 1) = 4;

        // Upstream gradient: shape (1, 1, 2, 2).
        // Every output position has gradient 1.
        ::Tensor grad_output(1, 1, 2, 2);
        grad_output.fill(1.0);


        // Test 1: Kernel gradient values.
        ::Tensor grad_kernels =
            cnn::convolutionKernelGradient(
                input, kernels, grad_output
            );

        if (grad_kernels.shape != vector<size_t>({1, 1, 2, 2})) {
            throw runtime_error(
                "Test 1 failed: kernel gradient shape"
            );
        }

        double expected_kernel_gradient[2][2] = {
            {12, 16},
            {24, 28}
        };

        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                if (!approximatelyEqual(
                    grad_kernels(0, 0, i, j),
                    expected_kernel_gradient[i][j]
                )) {
                    throw runtime_error(
                        "Test 1 failed: incorrect kernel gradient"
                    );
                }
            }
        }

        cout << "Test 1 passed: kernel gradients calculated correctly\n";


        // Test 2: Input gradient values.
        ::Tensor grad_input =
            cnn::convolutionInputGradient(
                input, kernels, grad_output
            );

        if (grad_input.shape != vector<size_t>({1, 1, 3, 3})) {
            throw runtime_error(
                "Test 2 failed: input gradient shape"
            );
        }

        double expected_input_gradient[3][3] = {
            {1, 3, 2},
            {4, 10, 6},
            {3, 7, 4}
        };

        for (size_t h = 0; h < 3; ++h) {
            for (size_t w = 0; w < 3; ++w) {
                if (!approximatelyEqual(
                    grad_input(0, 0, h, w),
                    expected_input_gradient[h][w]
                )) {
                    throw runtime_error(
                        "Test 2 failed: incorrect input gradient"
                    );
                }
            }
        }

        cout << "Test 2 passed: input gradients calculated correctly\n";


        // Test 3: Reject an incorrectly shaped upstream gradient.
        bool exception_thrown = false;

        try {
            ::Tensor wrong_gradient(1, 1, 3, 3);

            cnn::convolutionKernelGradient(
                input, kernels, wrong_gradient
            );
        }
        catch (const invalid_argument&) {
            exception_thrown = true;
        }

        if (!exception_thrown) {
            throw runtime_error(
                "Test 3 failed: invalid gradient shape accepted"
            );
        }

        cout << "Test 3 passed: invalid gradient shape rejected\n";


        // Test 4: Reject a kernel with incompatible input channels.
        exception_thrown = false;

        try {
            ::Tensor wrong_kernels(1, 2, 2, 2);
            wrong_kernels.fill(1.0);

            cnn::convolutionInputGradient(
                input, wrong_kernels, grad_output
            );
        }
        catch (const invalid_argument&) {
            exception_thrown = true;
        }

        if (!exception_thrown) {
            throw runtime_error(
                "Test 4 failed: incompatible channels accepted"
            );
        }

        cout << "Test 4 passed: incompatible channels rejected\n";


        cout << "\nAll 4 convolution backward tests passed.\n";
    }
    catch (const exception& e) {
        cerr << e.what() << '\n';
        return 1;
    }

    return 0;
}
