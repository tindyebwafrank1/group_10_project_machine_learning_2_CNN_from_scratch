
#include <iostream>
#include <stdexcept>
#include "convolution.hpp"

using namespace std;

int main() {
    int passed = 0;

    try {
        // Test 1: Basic 2D convolution
        Tensor input(1, 1, 3, 3);

        double values[3][3] = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };

        for (size_t h = 0; h < 3; ++h) {
            for (size_t w = 0; w < 3; ++w) {
                input(0, 0, h, w) = values[h][w];
            }
        }

        Tensor kernel(1, 1, 2, 2);
        kernel(0, 0, 0, 0) = 1;
        kernel(0, 0, 0, 1) = 0;
        kernel(0, 0, 1, 0) = 0;
        kernel(0, 0, 1, 1) = 1;

        Tensor output = cnn::conv2d_single(input, kernel);

        if (output.shape != vector<size_t>({1, 1, 2, 2})) {
            throw runtime_error("Test 1 failed: incorrect output shape");
        }

        double expected[2][2] = {
            {6, 8},
            {12, 14}
        };

        for (size_t h = 0; h < 2; ++h) {
            for (size_t w = 0; w < 2; ++w) {
                if (output(0, 0, h, w) != expected[h][w]) {
                    throw runtime_error("Test 1 failed: incorrect output values");
                }
            }
        }

        cout << "Test 1 passed: basic convolution\n";
        ++passed;

        // Test 2: Output dimension calculation
        if (cnn::conv_output_dim(5, 3) != 3) {
            throw runtime_error("Test 2 failed: incorrect output dimension");
        }

        cout << "Test 2 passed: output dimension\n";
        ++passed;

        // Test 3: Kernel larger than input
        bool exception_thrown = false;

        try {
            cnn::conv_output_dim(2, 3);
        } catch (const invalid_argument&) {
            exception_thrown = true;
        }

        if (!exception_thrown) {
            throw runtime_error("Test 3 failed: expected invalid_argument");
        }

        cout << "Test 3 passed: oversized kernel rejected\n";
        ++passed;

        // Test 4: Incorrect tensor shape
        exception_thrown = false;

        try {
            Tensor wrong_input(1, 2, 3, 3);
            cnn::conv2d_single(wrong_input, kernel);
        } catch (const invalid_argument&) {
            exception_thrown = true;
        }

        if (!exception_thrown) {
            throw runtime_error("Test 4 failed: expected invalid_argument");
        }

        cout << "Test 4 passed: invalid input shape rejected\n";
        ++passed;

        cout << "\nAll " << passed
             << " convolution tests passed successfully.\n";
    } catch (const exception& e) {
        cerr << e.what() << '\n';
        return 1;
    }

    return 0;
}
