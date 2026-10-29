
#include <iostream>
#include <stdexcept>
#include "../include/tensor.hpp"
#include "../include/convolution.hpp"

using namespace std;

int main()
{
    try
    {
        // Step 1: Create a 4x4 single-channel input tensor.
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

        // Step 2: Create a 2x2 convolution kernel.
        Tensor kernel(1, 1, 2, 2);

        kernel(0, 0, 0, 0) = 1;
        kernel(0, 0, 0, 1) = 0;
        kernel(0, 0, 1, 0) = 0;
        kernel(0, 0, 1, 1) = 1;

        // Step 3: Run the existing convolution implementation.
        Tensor output = cnn::conv2d_single(input, kernel);

        // Step 4: Display the input and kernel.
        cout << "GROUP 10 CNN - WEEK 1 DEMO\n";
        cout << "Convolution Pipeline Integration\n\n";

        cout << "Input tensor (4 x 4):\n";

        for (size_t h = 0; h < 4; h++)
        {
            for (size_t w = 0; w < 4; w++)
            {
                cout << input(0, 0, h, w) << "\t";
            }
            cout << "\n";
        }

        cout << "\nKernel (2 x 2):\n";

        for (size_t h = 0; h < 2; h++)
        {
            for (size_t w = 0; w < 2; w++)
            {
                cout << kernel(0, 0, h, w) << "\t";
            }
            cout << "\n";
        }

        // Step 5: Display the feature map.
        cout << "\nOutput feature map ("
             << output.shape[2] << " x "
             << output.shape[3] << "):\n";

        for (size_t h = 0; h < output.shape[2]; h++)
        {
            for (size_t w = 0; w < output.shape[3]; w++)
            {
                cout << output(0, 0, h, w) << "\t";
            }
            cout << "\n";
        }

        // Step 6: Verify the expected output dimensions.
        if (output.shape[2] != 3 || output.shape[3] != 3)
        {
            cerr << "\nERROR: Unexpected output dimensions.\n";
            return 1;
        }

        // Verify selected output values.
        if (output(0, 0, 0, 0) != 7 ||
            output(0, 0, 1, 1) != 17 ||
            output(0, 0, 2, 2) != 27)
        {
            cerr << "\nERROR: Convolution result is incorrect.\n";
            return 1;
        }

        cout << "\nSUCCESS: Convolution pipeline demo passed.\n";
        return 0;
    }
    catch (const exception& e)
    {
        cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
