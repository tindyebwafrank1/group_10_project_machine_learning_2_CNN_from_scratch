
#include <iostream>
#include <vector>
#include <exception>

#include "../include/tensor.hpp"
#include "../include/fully_connected.hpp"

using namespace std;

int main() {
    try {
        // Create input tensor with 2 samples and 4 features per sample.
        ::Tensor input(2, 1, 2, 2);

        input.data = {
            1, 2, 3, 4,
            4, 3, 2, 1
        };

        // Create a fully connected layer with 4 inputs and 2 outputs.
        cnn::FullyConnected layer(4, 2);

        // Assign weights.
        layer.setWeights({
            1, 0, 0, 0,
            0, 0, 0, 1
        });

        // Assign biases.
        layer.setBias({0, 0});

        // Forward propagation.
        ::Tensor output = layer.forward(input);

        cout << "FULLY CONNECTED LAYER DEMO\n";
        cout << "==========================\n";

        cout << "Forward propagation results:\n";

        for (size_t n = 0; n < 2; n++) {
            cout << "Sample " << n + 1 << ": ";

            for (size_t o = 0; o < 2; o++) {
                cout << output.data[n * 2 + o] << " ";
            }

            cout << endl;
        }

        // Check input caching.
        cout << "\nInput cached: "
             << (layer.hasCachedInput() ? "Yes" : "No")
             << endl;

        // Inference.
        ::Tensor inference = layer.infer(input);

        cout << "Inference successful: "
             << inference.data.size()
             << " outputs" << endl;

        // Clear cache.
        layer.clearCache();

        cout << "Cache cleared: "
             << (!layer.hasCachedInput() ? "Yes" : "No")
             << endl;

        cout << "\nDemo completed successfully.\n";

    }
    catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }

    return 0;
}
