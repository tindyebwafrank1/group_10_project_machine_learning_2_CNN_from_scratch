
#include "sgd_optimizer.hpp"

#include <stdexcept>
#include <cmath>

using namespace std;

namespace cnn {

SGDOptimizer::SGDOptimizer(double learning_rate)
    : learning_rate(learning_rate) {

    if (!isfinite(learning_rate) || learning_rate <= 0.0) {
        throw invalid_argument(
            "Learning rate must be finite and greater than zero."
        );
    }
}

void SGDOptimizer::step(
    vector<double>& parameters,
    const vector<double>& gradients) const {

    if (parameters.size() != gradients.size()) {
        throw invalid_argument(
            "Parameters and gradients must have equal sizes."
        );
    }

    // Validate first so invalid gradients cannot partially
    // update the parameter vector.
    for (size_t i = 0; i < gradients.size(); i++) {
        if (!isfinite(gradients[i])) {
            throw invalid_argument(
                "Gradients must contain only finite values."
            );
        }
    }

    // SGD update: parameter = parameter - learning_rate * gradient.
    for (size_t i = 0; i < parameters.size(); i++) {
        double updated_parameter =
            parameters[i] - learning_rate * gradients[i];

        if (!isfinite(updated_parameter)) {
            throw overflow_error(
                "SGD update produced a non-finite parameter."
            );
        }

        parameters[i] = updated_parameter;
    }
}

double SGDOptimizer::getLearningRate() const {
    return learning_rate;
}

void SGDOptimizer::setLearningRate(double new_learning_rate) {
    if (!isfinite(new_learning_rate) || new_learning_rate <= 0.0) {
        throw invalid_argument(
            "Learning rate must be finite and greater than zero."
        );
    }

    learning_rate = new_learning_rate;
}

}
