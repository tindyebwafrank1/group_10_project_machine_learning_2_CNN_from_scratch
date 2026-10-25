#include "metrics.hpp"

#include <iostream>
#include <vector>

using namespace std;

int main() {

    // Actual class labels from the dataset.
    vector<int> actual = {
        0, 0, 0, 1, 1, 1, 2, 2, 2, 2
    };

    // Predicted class labels from the argmax module.
    vector<int> predicted = {
        0, 0, 1, 1, 1, 2, 2, 2, 2, 0
    };

    // Calculate and display accuracy.
    cout << "Accuracy: "
         << cnn::accuracy(actual, predicted) << "\n";

    // Calculate precision for each class.
    cout << "Precision for class 0: "
         << cnn::precision(actual, predicted, 0) << "\n";

    cout << "Precision for class 1: "
         << cnn::precision(actual, predicted, 1) << "\n";

    cout << "Precision for class 2: "
         << cnn::precision(actual, predicted, 2) << "\n";

    // Calculate macro-averaged precision.
    cout << "Macro Precision: "
         << cnn::macroPrecision(actual, predicted) << "\n";

    return 0;
}
