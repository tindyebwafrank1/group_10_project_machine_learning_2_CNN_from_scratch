
#ifndef CNN_SGD_OPTIMIZER_HPP
#define CNN_SGD_OPTIMIZER_HPP

#include <vector>

using namespace std;

namespace cnn {

class SGDOptimizer {
private:
    double learning_rate;

public:
    explicit SGDOptimizer(double learning_rate = 0.01);

    // Update parameters using their gradients.
    void step(vector<double>& parameters,
              const vector<double>& gradients) const;

    double getLearningRate() const;

    void setLearningRate(double new_learning_rate);
};

}

#endif
