
#include "prediction.hpp"

#include <cmath>
#include <stdexcept>

using namespace std;

namespace cnn {

size_t argmax(const vector<double>& scores)
{
    // A prediction requires at least one class score.
    if (scores.empty()) {
        throw invalid_argument(
            "argmax: scores vector must not be empty"
        );
    }

    // Reject invalid numerical values.
    for (double score : scores) {
        if (!isfinite(score)) {
            throw invalid_argument(
                "argmax: scores must all be finite"
            );
        }
    }

    // Start with the first class as the best prediction.
    size_t best_index = 0;

    // Search for a score greater than the current maximum.
    for (size_t i = 1; i < scores.size(); ++i) {
        if (scores[i] > scores[best_index]) {
            best_index = i;
        }
    }

    return best_index;
}

size_t predict(const vector<double>& scores)
{
    return argmax(scores);
}

} // namespace cnn
