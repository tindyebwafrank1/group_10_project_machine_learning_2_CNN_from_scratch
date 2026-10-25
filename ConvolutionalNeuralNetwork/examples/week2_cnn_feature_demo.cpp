
#include <iostream>
#include <stdexcept>

#include "tensor.hpp"
#include "convolution.hpp"
#include "relu.hpp"
#include "max_pooling.hpp"

using namespace std;

int main()
{
    try
    {
        cout << "GROUP 10 CNN - WEEK 2 DEMO\n";
        cout << "CNN Feature Integration\n\n";

        // Step 1: Create a single-channel 4x4 input tensor.
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

        cout << "Input shape: ("
             << input.shape[0] << ", "
             << input.shape[1] << ", "
             << input.shape[2] << ", "
             << input.shape[3] << ")\n";

        // Step 2: Create a 2x2 convolution kernel.
        Tensor kernel(1, 1, 2, 2);

        kernel(0, 0, 0, 0) =  1.0;
        kernel(0, 0, 0, 1) =  0.0;
        kernel(0, 0, 1, 0) =  0.0;
        kernel(0, 0, 1, 1) = -1.0;

        // Step 3: Perform valid 2D convolution.
        Tensor conv_output = cnn::conv2d_single(input, kernel);

        cout << "\nAfter convolution: ("
             << conv_output.shape[0] << ", "
             << conv_output.shape[1] << ", "
             << conv_output.shape[2] << ", "
             << conv_output.shape[3] << ")\n";

        // Step 4: Apply ReLU activation.
        Tensor relu_output = cnn::relu(conv_output);

        cout << "\nAfter ReLU:\n";

        for (size_t h = 0; h < relu_output.shape[2]; h++)
        {
            for (size_t w = 0; w < relu_output.shape[3]; w++)
            {
                cout << relu_output(0, 0, h, w) << "\t";
            }
            cout << "\n";
        }

        // Step 5: Apply overlapping 2x2 max pooling.
        PoolingParams params;
        params.window_h = 2;
        params.window_w = 2;
        params.stride_h = 1;
        params.stride_w = 1;

        MaxPoolingResult pool_result = maxPool2D(relu_output, params);
        Tensor pooled_output = pool_result.output;

        cout << "\nAfter max pooling: ("
             << pooled_output.shape[0] << ", "
             << pooled_output.shape[1] << ", "
             << pooled_output.shape[2] << ", "
             << pooled_output.shape[3] << ")\n";

        for (size_t h = 0; h < pooled_output.shape[2]; h++)
        {
            for (size_t w = 0; w < pooled_output.shape[3]; w++)
            {
                cout << pooled_output(0, 0, h, w) << "\t";
            }
            cout << "\n";
        }

        // Step 6: Verify each stage's expected dimensions.
        if (conv_output.shape[2] != 3 ||
            conv_output.shape[3] != 3)
        {
            throw runtime_error("Unexpected convolution output dimensions.");
        }

        if (relu_output.shape != conv_output.shape)
        {
            throw runtime_error("ReLU changed the tensor dimensions.");
        }

        if (pooled_output.shape[2] != 2 ||
            pooled_output.shape[3] != 2)
        {
            throw runtime_error("Unexpected max-pooling output dimensions.");
        }

        // Verify the expected pooled feature map.
        if (pooled_output(0, 0, 0, 0) != 0.0 ||
            pooled_output(0, 0, 0, 1) != 1.0 ||
            pooled_output(0, 0, 1, 0) != 0.0 ||
            pooled_output(0, 0, 1, 1) != 1.0)
        {
            throw runtime_error("Feature-map values are incorrect.");
        }

        cout << "\nSUCCESS: Week 2 CNN feature integration passed.\n";
        return 0;
    }
    catch (const exception& e)
    {
        cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
}
