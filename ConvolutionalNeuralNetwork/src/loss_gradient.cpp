#include "loss_gradient.hpp"

#include <stdexcept>
#include <vector>

using namespace std;

namespace cnn {

// Calculate the gradient of cross-entropy loss with respect to the Softmax input logits.
// Formula: dL/dz = probabilities - targets
vector<double> softmaxCrossEntropyGradient(
    const vector<double>& probabilities,
    const vector<double>& targets)
{
    // Check that both vectors have equal, non-zero sizes.
    if (probabilities.empty() ||
        probabilities.size() != targets.size()) {
        throw invalid_argument(
            "Probabilities and targets must have equal, non-zero sizes");
    }

    // Create a vector to store the gradients.
    vector<double> gradient(probabilities.size());

    // Calculate the gradient for each output class.
    for (size_t i = 0; i < probabilities.size(); ++i) {

        // Validate probabilities and target values.
        if (probabilities[i] < 0.0 ||
            probabilities[i] > 1.0 ||
            targets[i] < 0.0 ||
            targets[i] > 1.0) {
            throw invalid_argument(
                "Probabilities and targets must be in [0, 1]");
        }

        // Subtract the target from the predicted probability.
        gradient[i] = probabilities[i] - targets[i];
    }

    // Return the calculated gradients.
    return gradient;
}

// Backpropagate a general gradient through Softmax.
// Formula: dz_i = p_i * (g_i - sum(p_j * g_j))
vector<double> softmaxBackward(
    const vector<double>& probabilities,
    const vector<double>& upstream_gradient)
{
    // Check that both vectors have equal, non-zero sizes.
    if (probabilities.empty() ||
        probabilities.size() != upstream_gradient.size()) {
        throw invalid_argument(
            "Probabilities and upstream gradient sizes must match");
    }

    // Calculate the dot product of probabilities and upstream gradients.
    double dot_product = 0.0;

    for (size_t i = 0; i < probabilities.size(); ++i) {
        dot_product += probabilities[i] * upstream_gradient[i];
    }

    // Create a vector to store the Softmax gradients.
    vector<double> gradient(probabilities.size());

    // Apply the Softmax backward derivative.
    for (size_t i = 0; i < probabilities.size(); ++i) {
        gradient[i] = probabilities[i] * (upstream_gradient[i] - dot_product);
    }

    return gradient;
}

} 