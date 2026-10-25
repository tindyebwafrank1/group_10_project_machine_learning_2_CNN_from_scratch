
#include "metrics.hpp"

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>

using namespace std;

int main() {

    vector<int> actual = {0, 1, 1, 0, 2, 2};
    vector<int> predicted = {0, 1, 0, 0, 2, 1};

    // Test accuracy: 4 correct predictions out of 6.
    double acc = cnn::accuracy(actual, predicted);

    if (abs(acc - (4.0 / 6.0)) > 1e-9) {
        cerr << "Accuracy test failed.\n";
        return 1;
    }

    // Class 0: 2 true positives out of 3 predictions.
    double p0 = cnn::precision(actual, predicted, 0);

    if (abs(p0 - (2.0 / 3.0)) > 1e-9) {
        cerr << "Class 0 precision test failed.\n";
        return 1;
    }

    // Class 1: 1 true positive out of 2 predictions.
    double p1 = cnn::precision(actual, predicted, 1);

    if (abs(p1 - 0.5) > 1e-9) {
        cerr << "Class 1 precision test failed.\n";
        return 1;
    }

    // Class 2: 1 true positive out of 1 prediction.
    double p2 = cnn::precision(actual, predicted, 2);

    if (abs(p2 - 1.0) > 1e-9) {
        cerr << "Class 2 precision test failed.\n";
        return 1;
    }

    // Macro precision = (2/3 + 1/2 + 1) / 3.
    double mp = cnn::macroPrecision(actual, predicted);

    if (abs(mp - (13.0 / 18.0)) > 1e-9) {
        cerr << "Macro precision test failed.\n";
        return 1;
    }

    // Test mismatched input sizes.
    try {
        vector<int> wrong = {0, 1};

        cnn::accuracy(actual, wrong);

        cerr << "Mismatched-size test failed.\n";
        return 1;

    } catch (const invalid_argument&) {
        // Expected exception.
    }

    // Test empty input.
    try {
        vector<int> empty;

        cnn::accuracy(empty, empty);

        cerr << "Empty-input test failed.\n";
        return 1;

    } catch (const invalid_argument&) {
        // Expected exception.
    }

    cout << "All metrics tests passed successfully.\n";

    return 0;
}
