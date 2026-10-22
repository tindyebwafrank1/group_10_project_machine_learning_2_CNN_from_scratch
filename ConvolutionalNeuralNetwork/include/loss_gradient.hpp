#ifndef CNN_LOSS_GRADIENT_HPP
#define CNN_LOSS_GRADIENT_HPP

#include <vector>

using namespace std;

namespace cnn {

// Gradient of categorical cross-entropy loss
// with respect to the Softmax input logits.
vector<double> softmaxCrossEntropyGradient(
    const vector<double>& probabilities,
    const vector<double>& targets
);

// Backpropagation through Softmax for an arbitrary
// upstream gradient.
vector<double> softmaxBackward(
    const vector<double>& probabilities,
    const vector<double>& upstream_gradient
);

} // namespace cnn

#endif