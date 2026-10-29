
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../include/tensor.hpp"
#include "../include/multi_channel_convolution.hpp"

using namespace std;

// SYSTEM INTEGRATION AND DEMONSTRATION

class PipelineIntegrationManager {
public:

    // Test that Tensor detects invalid coordinates.
    static bool testTensorBounds() {
        cout << "[Test 1] Tensor boundary protection\n";

        Tensor input(1, 1, 4, 4);

        try {
            input(1, 0, 0, 0);  // Invalid batch index
            cout << "FAILED: Invalid index was accepted.\n";
            return false;
        }
        catch (const out_of_range&) {
            cout << "PASSED: Invalid index was rejected.\n";
        }

        return true;
    }

    // Test convolution using a small, predictable example.
    static bool testConvolution() {
        cout << "\n[Test 2] Multi-channel convolution\n";

        Tensor input(1, 1, 3, 3);
        Tensor filters(1, 1, 2, 2);

        // Fill the input with ones.
        fill(input.data.begin(), input.data.end(), 1.0);

        // Fill the filter with twos.
        fill(filters.data.begin(), filters.data.end(), 2.0);

        // Perform valid convolution/cross-correlation.
        Tensor output = cnn::multiChannelConvolution(input, filters);

        // Each output value should be:
        // 1*2 + 1*2 + 1*2 + 1*2 = 8
        if (output.shape.size() != 4 ||
            output.shape[0] != 1 ||
            output.shape[1] != 1 ||
            output.shape[2] != 2 ||
            output.shape[3] != 2) {
            cout << "FAILED: Unexpected output shape.\n";
            return false;
        }

        for (double value : output.data) {
            if (abs(value - 8.0) > 1e-9) {
                cout << "FAILED: Incorrect convolution result.\n";
                return false;
            }
        }

        cout << "PASSED: Output shape is (1, 1, 2, 2).\n";
        cout << "PASSED: All output values equal 8.\n";

        return true;
    }

    // Print the tensor dimensions.
    static void displayTensorShape(
        const string& label,
        const Tensor& tensor
    ) {
        cout << label << ": (";

        for (size_t i = 0; i < tensor.shape.size(); ++i) {
            cout << tensor.shape[i];

            if (i + 1 < tensor.shape.size()) {
                cout << ", ";
            }
        }

        cout << ")\n";
    }
};

int main() {
    cout << "\n";
    cout << " CNN SYSTEM INTEGRATION DEMONSTRATION\n";
    cout << "\n\n";

    try {
        bool boundsPassed =
            PipelineIntegrationManager::testTensorBounds();

        bool convolutionPassed =
            PipelineIntegrationManager::testConvolution();

        cout << "\n VALIDATION SUMMARY \n";

        cout << "Tensor bounds test: "
             << (boundsPassed ? "PASSED" : "FAILED") << '\n';

        cout << "Convolution test: "
             << (convolutionPassed ? "PASSED" : "FAILED") << '\n';

        if (boundsPassed && convolutionPassed) {
            cout << "\nAll integration demonstration tests passed.\n";
            return 0;
        }

        cout << "\nSome tests failed. Check the relevant modules.\n";
        return 1;
    }
    catch (const exception& error) {
        cerr << "\nIntegration error: "
             << error.what() << '\n';

        return 1;
    }
}
