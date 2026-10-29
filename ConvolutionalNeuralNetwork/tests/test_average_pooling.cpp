
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>

#include "tensor.hpp"
#include "pooling.hpp"

using namespace std;

int main()
{
    try
    {
        // Basic average pooling.
        Tensor input(1, 1, 4, 4);

        double values[4][4] = {
            {1, 3, 5, 7},
            {5, 7, 9, 11},
            {2, 4, 6, 8},
            {6, 8, 10, 12}
        };

        for (size_t h = 0; h < 4; ++h)
        {
            for (size_t w = 0; w < 4; ++w)
            {
                input(0, 0, h, w) = values[h][w];
            }
        }

        cnn::PoolingParams params;
        params.window_h = 2;
        params.window_w = 2;
        params.stride_h = 2;
        params.stride_w = 2;

        cnn::AveragePooling2D pool(params);
        Tensor output = pool.forward(input);

        if (output.shape != vector<size_t>({1, 1, 2, 2}))
        {
            throw runtime_error("Test 1 failed: incorrect output shape");
        }

        double expected[2][2] = {
            {4.0, 8.0},
            {5.0, 9.0}
        };

        for (size_t h = 0; h < 2; ++h)
        {
            for (size_t w = 0; w < 2; ++w)
            {
                if (fabs(output(0, 0, h, w) - expected[h][w]) > 1e-9)
                {
                    throw runtime_error(
                        "Test 1 failed: incorrect average"
                    );
                }
            }
        }

        cout << "Test 1 passed: average pooling values and shape\n";

        // A window of size 1x1 preserves values.
        cnn::PoolingParams single_params;
        single_params.window_h = 1;
        single_params.window_w = 1;
        single_params.stride_h = 1;
        single_params.stride_w = 1;

        cnn::AveragePooling2D single_pool(single_params);
        Tensor single_output = single_pool.forward(input);

        for (size_t h = 0; h < 4; ++h)
        {
            for (size_t w = 0; w < 4; ++w)
            {
                if (fabs(
                    single_output(0, 0, h, w) - input(0, 0, h, w)
                ) > 1e-9)
                {
                    throw runtime_error(
                        "Test 2 failed: 1x1 pooling changed a value"
                    );
                }
            }
        }

        cout << "Test 2 passed: 1x1 pooling preserves values\n";

        // Zero window size must be rejected.
        bool exception_thrown = false;

        try
        {
            cnn::PoolingParams invalid_params;
            invalid_params.window_h = 0;

            cnn::AveragePooling2D invalid_pool(invalid_params);
            invalid_pool.forward(input);
        }
        catch (const invalid_argument&)
        {
            exception_thrown = true;
        }

        if (!exception_thrown)
        {
            throw runtime_error(
                "Test 3 failed: zero window size was accepted"
            );
        }

        cout << "Test 3 passed: zero window size rejected\n";

        //A window larger than the input must be rejected.
        exception_thrown = false;

        try
        {
            cnn::PoolingParams large_params;
            large_params.window_h = 5;
            large_params.window_w = 2;

            cnn::AveragePooling2D large_pool(large_params);
            large_pool.forward(input);
        }
        catch (const invalid_argument&)
        {
            exception_thrown = true;
        }

        if (!exception_thrown)
        {
            throw runtime_error(
                "Test 4 failed: oversized window was accepted"
            );
        }

        cout << "Test 4 passed: oversized window rejected\n";

        cout << "\nAll 4 average pooling tests passed.\n";
    }
    catch (const exception& e)
    {
        cerr << e.what() << '\n';
        return 1;
    }

    return 0;
}
