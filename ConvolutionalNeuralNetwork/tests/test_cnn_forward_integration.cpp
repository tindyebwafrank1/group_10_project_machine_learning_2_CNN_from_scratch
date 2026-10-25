
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "tensor.hpp"
#include "convolution.hpp"
#include "relu.hpp"
#include "max_pooling.hpp"
#include "week3_forward_utils.hpp"

using namespace std;

int main()
{
    try
    {
        // Step 1: Prepare the input tensor.
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

        // Step 2: Prepare the convolution kernel.
        Tensor kernel(1, 1, 2, 2);

        kernel(0, 0, 0, 0) =  1.0;
        kernel(0, 0, 0, 1) =  0.0;
        kernel(0, 0, 1, 0) =  0.0;
        kernel(0, 0, 1, 1) = -1.0;

        // Step 3: Perform convolution.
        Tensor conv_output = cnn::conv2d_single(input, kernel);

        if (conv_output.shape[2] != 3 ||
            conv_output.shape[3] != 3)
        {
            throw runtime_error(
                "Convolution output dimensions are incorrect.");
        }

        if (conv_output(0, 0, 0, 0) != 0.0 ||
    conv_output(0, 0, 0, 1) != -5.0 ||
    conv_output(0, 0, 2, 2) != 1.0)
{
    throw runtime_error(
        "Convolution output values are incorrect.");
}

        // Step 4: Apply ReLU activation.
        Tensor relu_output = cnn::relu(conv_output);

        if (relu_output.shape != conv_output.shape)
        {
            throw runtime_error(
                "ReLU changed the tensor dimensions.");
        }

        for (double value : relu_output.data)
        {
            if (value < 0.0)
            {
                throw runtime_error(
                    "ReLU produced a negative value.");
            }
        }

        if (relu_output(0, 0, 0, 1) != 0.0 ||
    relu_output(0, 0, 0, 2) != 1.0 ||
    relu_output(0, 0, 2, 2) != 1.0)
{
    throw runtime_error(
        "ReLU output values are incorrect.");
}

        // Step 5: Apply max pooling.
        PoolingParams params;
        params.window_h = 2;
        params.window_w = 2;
        params.stride_h = 1;
        params.stride_w = 1;

        MaxPoolingResult pool_result =
            maxPool2D(relu_output, params);

        Tensor pooled_output = pool_result.output;

        if (pooled_output.shape[2] != 2 ||
            pooled_output.shape[3] != 2)
        {
            throw runtime_error(
                "Max-pooling output dimensions are incorrect.");
        }

        if (pooled_output(0, 0, 0, 0) != 0.0 ||
            pooled_output(0, 0, 0, 1) != 1.0 ||
            pooled_output(0, 0, 1, 0) != 0.0 ||
            pooled_output(0, 0, 1, 1) != 1.0)
        {
            throw runtime_error(
                "Max-pooling output values are incorrect.");
        }

        // Step 6: Flatten the pooled feature map.
        vector<double> features =
            week3_app::flattenTensor(pooled_output);

        if (features.size() != 4)
        {
            throw runtime_error(
                "Flattening produced an incorrect feature count.");
        }

        // Step 7: Fully connected layer.
        vector<vector<double>> weights = {
            {0.1, 0.2, 0.1, 0.2},
            {0.5, 0.1, 0.3, 0.2},
            {0.1, 0.6, 0.2, 0.7}
        };

        vector<double> biases = {0.1, 0.0, 0.2};

        vector<double> logits =
            week3_app::fullyConnected(features, weights, biases);

        if (logits.size() != 3)
        {
            throw runtime_error(
                "Fully connected layer should produce three logits.");
        }

        // Step 8: Calculate softmax probabilities.
        vector<double> probabilities =
            week3_app::softmax(logits);

        if (probabilities.size() != 3)
        {
            throw runtime_error(
                "Softmax returned an incorrect number of probabilities.");
        }

        double probability_sum = 0.0;

        for (double probability : probabilities)
        {
            if (!isfinite(probability) ||
                probability < 0.0 ||
                probability > 1.0)
            {
                throw runtime_error(
                    "Softmax produced an invalid probability.");
            }

            probability_sum += probability;
        }

        if (abs(probability_sum - 1.0) > 1e-9)
        {
            throw runtime_error(
                "Softmax probabilities do not sum to one.");
        }

        // Step 9: Predict the class using argmax.
        size_t prediction =
            week3_app::argmax(probabilities);

        if (prediction >= probabilities.size())
        {
            throw runtime_error(
                "Predicted class index is out of range.");
        }

        if (prediction != 2)
        {
            throw runtime_error(
                "Unexpected prediction for the fixed demo weights.");
        }

        cout << "All CNN forward integration tests passed!\n";
        cout << "Predicted demo class: " << prediction << "\n";

        return 0;
    }
    catch (const exception& e)
    {
        cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
