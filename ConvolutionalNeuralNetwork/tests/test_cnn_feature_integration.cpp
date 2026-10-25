
#include <iostream>
#include <stdexcept>

#include "tensor.hpp"
#include "convolution.hpp"
#include "relu.hpp"
#include "max_pooling.hpp"

using namespace std;

int main()
{
    int failures = 0;

    // Prepare the input.
    Tensor input(1, 1, 4, 4);

    double values[4][4] = {
        {1,  2,  3,  4},
        {5,  1,  7,  2},
        {9, 10,  2, 12},
        {3, 14, 15,  1}
    };

    for (size_t h = 0; h < 4; h++)
    {
        for (size_t w = 0; w < 4; w++)
        {
            input(0, 0, h, w) = values[h][w];
        }
    }

    // Prepare the convolution kernel.
    Tensor kernel(1, 1, 2, 2);
    kernel(0, 0, 0, 0) =  1.0;
    kernel(0, 0, 0, 1) =  0.0;
    kernel(0, 0, 1, 0) =  0.0;
    kernel(0, 0, 1, 1) = -1.0;

    // Test convolution.
    Tensor conv_output = cnn::conv2d_single(input, kernel);

    if (conv_output.shape[2] != 3 ||
        conv_output.shape[3] != 3)
    {
        cerr << "FAILED: Convolution output dimensions\n";
        failures++;
    }

    if (conv_output(0, 0, 0, 0) != 0.0 ||
        conv_output(0, 0, 0, 1) != -5.0 ||
        conv_output(0, 0, 2, 2) != 1.0)
    {
        cerr << "FAILED: Convolution output values\n";
        failures++;
    }

    // Test ReLU.
    Tensor relu_output = cnn::relu(conv_output);

    if (relu_output.shape != conv_output.shape)
    {
        cerr << "FAILED: ReLU output dimensions\n";
        failures++;
    }

    for (size_t i = 0; i < relu_output.data.size(); i++)
    {
        if (relu_output.data[i] < 0.0)
        {
            cerr << "FAILED: ReLU left a negative value\n";
            failures++;
            break;
        }
    }

    if (relu_output(0, 0, 0, 1) != 0.0 ||
        relu_output(0, 0, 0, 2) != 1.0)
    {
        cerr << "FAILED: ReLU output values\n";
        failures++;
    }

    // Test max pooling.
    PoolingParams params;
    params.window_h = 2;
    params.window_w = 2;
    params.stride_h = 1;
    params.stride_w = 1;

    MaxPoolingResult pool_result = maxPool2D(relu_output, params);
    Tensor pooled_output = pool_result.output;

    if (pooled_output.shape[2] != 2 ||
        pooled_output.shape[3] != 2)
    {
        cerr << "FAILED: Max-pooling output dimensions\n";
        failures++;
    }

    if (pooled_output(0, 0, 0, 0) != 0.0 ||
        pooled_output(0, 0, 0, 1) != 1.0 ||
        pooled_output(0, 0, 1, 0) != 0.0 ||
        pooled_output(0, 0, 1, 1) != 1.0)
    {
        cerr << "FAILED: Max-pooling output values\n";
        failures++;
    }

    if (failures == 0)
    {
        cout << "All CNN feature integration tests passed!\n";
        return 0;
    }

    cerr << failures << " integration test(s) failed.\n";
    return 1;
}
