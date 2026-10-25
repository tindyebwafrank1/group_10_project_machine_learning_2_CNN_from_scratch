#include "weight_initialization.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

using namespace std;

namespace cnn {

vector<double> initializeWeights(
    size_t fan_in,
    size_t fan_out,
    InitializationMethod method
) {
    if (fan_in == 0 || fan_out == 0) {
        throw invalid_argument(
            "fan_in and fan_out must be greater than zero."
        );
    }

    // A random generator for this initialization call.
    random_device rd;
    mt19937 generator(rd());

    vector<double> weights(fan_in * fan_out);

    if (method == InitializationMethod::Xavier) {
        const double limit = sqrt(
            6.0 / static_cast<double>(fan_in + fan_out)
        );

        uniform_real_distribution<double> distribution(
            -limit, limit
        );

        for (double& weight : weights) {
            weight = distribution(generator);
        }
    }
    else if (method == InitializationMethod::He) {
        const double standard_deviation = sqrt(
            2.0 / static_cast<double>(fan_in)
        );

        normal_distribution<double> distribution(
            0.0, standard_deviation
        );

        for (double& weight : weights) {
            weight = distribution(generator);
        }
    }
    else {
        throw invalid_argument(
            "Unknown weight initialization method."
        );
    }

    return weights;
}

vector<double> initializeBiases(size_t count) {
    return vector<double>(count, 0.0);
}

} // namespace cnn