
#include "metrics.hpp"

#include <stdexcept>
#include <set>

using namespace std;

namespace cnn {

namespace {

// Validate inputs before calculating any metric.
void validateInputs(const vector<int>& actual,
                    const vector<int>& predicted) {

    if (actual.empty()) {
        throw invalid_argument(
            "Actual labels cannot be empty."
        );
    }

    if (actual.size() != predicted.size()) {
        throw invalid_argument(
            "Actual and predicted labels must have equal sizes."
        );
    }
}

}

// Calculate accuracy = correct predictions / total predictions.
double accuracy(const vector<int>& actual,
                const vector<int>& predicted) {

    validateInputs(actual, predicted);

    size_t correct = 0;

    for (size_t i = 0; i < actual.size(); i++) {
        if (actual[i] == predicted[i]) {
            correct++;
        }
    }

    return static_cast<double>(correct) / actual.size();
}

// Calculate precision for a specific class:
// TP / (TP + FP).
// Returns 0 when the class has no predicted positives.
double precision(const vector<int>& actual,
                 const vector<int>& predicted,
                 int class_label) {

    validateInputs(actual, predicted);

    size_t true_positives = 0;
    size_t false_positives = 0;

    for (size_t i = 0; i < actual.size(); i++) {

        if (predicted[i] == class_label) {

            if (actual[i] == class_label) {
                true_positives++;
            } else {
                false_positives++;
            }
        }
    }

    size_t predicted_positives =
        true_positives + false_positives;

    if (predicted_positives == 0) {
        return 0.0;
    }

    return static_cast<double>(true_positives)
           / predicted_positives;
}

// Calculate macro precision by averaging the precision of every distinct class found in either label vector.
double macroPrecision(const vector<int>& actual,
                      const vector<int>& predicted) {

    validateInputs(actual, predicted);

    set<int> classes;

    for (size_t i = 0; i < actual.size(); i++) {
        classes.insert(actual[i]);
        classes.insert(predicted[i]);
    }

    double total_precision = 0.0;

    for (int class_label : classes) {
        total_precision += precision(
            actual, predicted, class_label
        );
    }

    return total_precision / classes.size();
}

}
