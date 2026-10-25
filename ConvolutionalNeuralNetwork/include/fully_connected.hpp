
#ifndef CNN_FULLY_CONNECTED_HPP
#define CNN_FULLY_CONNECTED_HPP

#include <cstddef>
#include <vector>

using namespace std;

namespace cnn {

// Computes z = Wx + b for a single input vector.
//
// weights shape: [number of output neurons][number of input features]
// input shape:   [number of input features]
// biases shape:  [number of output neurons]
vector<double> fullyConnectedForward(
    const vector<vector<double>>& weights,
    const vector<double>& input,
    const vector<double>& biases
);

// Represents a fully connected (dense) layer.
class FullyConnectedLayer {
private:
    vector<vector<double>> weights;
    vector<double> biases;

public:
    FullyConnectedLayer(
        const vector<vector<double>>& layer_weights,
        const vector<double>& layer_biases
    );

    // Returns the pre-activation values z = Wx + b.
    vector<double> forward(const vector<double>& input) const;

    size_t inputSize() const;
    size_t outputSize() const;

    const vector<vector<double>>& getWeights() const;
    const vector<double>& getBiases() const;
};

} // namespace cnn

#endif
