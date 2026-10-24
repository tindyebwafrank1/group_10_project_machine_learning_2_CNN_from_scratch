// Demonstrates the metrics suite on hand-written predictions.
// In the full pipeline, `predicted` comes from the argmax prediction module (M6)
// and `actual` from the dataset labels.
#include <iostream>
#include "cnn_library/metrics.hpp"

int main() {
    std::vector<size_t> actual    = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    std::vector<size_t> predicted = {0, 0, 1, 1, 1, 2, 2, 2, 2, 0};

    CNN::ConfusionMatrix cm(3);
    cm.add_batch(actual, predicted);
    std::cout << cm.to_string();
    std::cout << "Quick accuracy: " << CNN::accuracy(actual, predicted) << "\n";
    return 0;
}
