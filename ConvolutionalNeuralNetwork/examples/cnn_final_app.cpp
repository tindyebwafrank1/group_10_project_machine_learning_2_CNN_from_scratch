#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

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
        cout << "GROUP 10 CNN - FINAL FORWARD APP\n";
        cout << "Week 3: Complete Forward CNN Integration\n\n";

        // 1. Create the sample input tensor.
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

        cout << "Input tensor shape: ";
        week3_app::printShape(input);

        // 2. Convolution.
        Tensor kernel(1, 1, 2, 2);

        kernel(0, 0, 0, 0) =  1.0;
        kernel(0, 0, 0, 1) =  0.0;
        kernel(0, 0, 1, 0) =  0.0;
        kernel(0, 0, 1, 1) = -1.0;

        Tensor conv_output = cnn::conv2d_single(input, kernel);

        cout << "\nConvolution output shape: ";
        week3_app::printShape(conv_output);

        // 3. ReLU activation.
        Tensor relu_output = cnn::relu(conv_output);

        cout << "\nReLU feature map:\n";
        week3_app::printMatrix(relu_output);

        // 4. Max pooling.
        PoolingParams params;
        params.window_h = 2;
        params.window_w = 2;
        params.stride_h = 1;
        params.stride_w = 1;

        MaxPoolingResult pool_result = maxPool2D(relu_output, params);
        Tensor pooled_output = pool_result.output;

        cout << "\nPooled feature map:\n";
        week3_app::printMatrix(pooled_output);

        // 5. Flatten the feature map.
        vector<double> features =
            week3_app::flattenTensor(pooled_output);

        cout << "\nFlattened features: ";
        for (double value : features)
        {
            cout << value << " ";
        }
        cout << "\n";

        // 6. Fully connected layer.
        // Each row contains the weights for one output class.
        vector<vector<double>> weights = {
            {0.1, 0.2, 0.1, 0.2},
            {0.5, 0.1, 0.3, 0.2},
            {0.1, 0.6, 0.2, 0.7}
        };

        vector<double> biases = {0.1, 0.0, 0.2};

        vector<double> logits =
            week3_app::fullyConnected(features, weights, biases);

        cout << "\nFully connected logits: ";
        for (double value : logits)
        {
            cout << value << " ";
        }
        cout << "\n";

        // 7. Softmax probabilities.
        vector<double> probabilities =
            week3_app::softmax(logits);

        cout << "\nClass probabilities:\n";
        for (size_t i = 0; i < probabilities.size(); i++)
        {
            cout << "Class " << i << ": "
                 << probabilities[i] << "\n";
        }

        // 8. Select the class with the highest probability.
        size_t predicted_class =
            week3_app::argmax(probabilities);

        cout << "\nPredicted class: "
             << predicted_class << "\n";

        // 9. Basic forward-pass verification.
        double probability_sum = 0.0;

        for (double p : probabilities)
        {
            if (!isfinite(p) || p < 0.0 || p > 1.0)
            {
                throw runtime_error(
                    "Invalid class probability produced.");
            }

            probability_sum += p;
        }

        if (abs(probability_sum - 1.0) > 1e-9)
        {
            throw runtime_error(
                "Softmax probabilities do not sum to one.");
        }

        if (predicted_class >= probabilities.size())
        {
            throw runtime_error(
                "Predicted class index is out of range.");
        }

        cout << "\nSUCCESS: Complete forward CNN demo passed.\n";
        cout << "Note: This is a demonstration using fixed weights, "
             << "not a trained classifier.\n";

        return 0;
    }
    catch (const exception& e)
    {
        cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
}