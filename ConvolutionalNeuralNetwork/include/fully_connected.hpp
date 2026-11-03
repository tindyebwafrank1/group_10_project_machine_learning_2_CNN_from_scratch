
#ifndef FULLY_CONNECTED_HPP
#define FULLY_CONNECTED_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <limits>
#include <algorithm>
#include <cstddef>

#include "tensor.hpp"

using namespace std;

namespace cnn {

class FullyConnected {
private:
    size_t in_features_;
    size_t out_features_;

    vector<double> weights_;
    vector<double> bias_;

    bool has_cache_ = false;
    ::Tensor cached_input_;

    void validateInput(const ::Tensor& input) const {
        if (input.shape.size() != 4) {
            throw invalid_argument(
                "FullyConnected: input must be a 4-D tensor {N,C,H,W}"
            );
        }

        size_t N = input.shape[0];
        size_t C = input.shape[1];
        size_t H = input.shape[2];
        size_t W = input.shape[3];

        if (C == 0 || H == 0 || W == 0) {
            throw invalid_argument("FullyConnected: input dimensions cannot be zero");
        }

        if (C > numeric_limits<size_t>::max() / H ||
            C * H > numeric_limits<size_t>::max() / W) {
            throw invalid_argument("FullyConnected: input dimensions overflow");
        }

        size_t features = C * H * W;

        if (features != in_features_) {
            throw invalid_argument(
                "FullyConnected: incorrect number of input features"
            );
        }

        if (N > numeric_limits<size_t>::max() / in_features_ ||
            input.data.size() != N * in_features_) {
            throw invalid_argument(
                "FullyConnected: input data size does not match shape"
            );
        }

        for (double value : input.data) {
            if (!isfinite(value)) {
                throw invalid_argument(
                    "FullyConnected: input contains NaN or Inf"
                );
            }
        }
    }

    ::Tensor computeForward(const ::Tensor& input) const {
        size_t N = input.shape[0];

        ::Tensor output(N, out_features_, 1, 1);

        for (size_t n = 0; n < N; n++) {
            for (size_t o = 0; o < out_features_; o++) {
                double sum = bias_[o];

                for (size_t i = 0; i < in_features_; i++) {
                    double x = input.data[n * in_features_ + i];
                    double w = weights_[o * in_features_ + i];

                    sum += w * x;

                    if (!isfinite(sum)) {
                        throw runtime_error(
                            "FullyConnected: numerical overflow during forward pass"
                        );
                    }
                }

                output.data[n * out_features_ + o] = sum;
            }
        }

        return output;
    }

public:
    FullyConnected(size_t in_features, size_t out_features)
        : in_features_(in_features), out_features_(out_features) {

        if (in_features == 0 || out_features == 0) {
            throw invalid_argument(
                "FullyConnected: feature dimensions must be positive"
            );
        }

        if (in_features > numeric_limits<size_t>::max() / out_features) {
            throw invalid_argument(
                "FullyConnected: weight dimensions are too large"
            );
        }

        weights_.assign(in_features * out_features, 0.0);
        bias_.assign(out_features, 0.0);
    }

    // Forward propagation
    ::Tensor forward(const ::Tensor& input) {
        validateInput(input);

        ::Tensor output = computeForward(input);

        cached_input_ = input;
        has_cache_ = true;

        return output;
    }

    // Inference without changing the cache
    ::Tensor infer(const ::Tensor& input) const {
        validateInput(input);
        return computeForward(input);
    }

    // Accessors
    size_t inFeatures() const {
        return in_features_;
    }

    size_t outFeatures() const {
        return out_features_;
    }

    size_t parameterCount() const {
        return weights_.size() + bias_.size();
    }

    const vector<double>& weights() const {
        return weights_;
    }

    const vector<double>& bias() const {
        return bias_;
    }

    double weight(size_t out_idx, size_t in_idx) const {
        if (out_idx >= out_features_ || in_idx >= in_features_) {
            throw out_of_range(
                "FullyConnected::weight index out of range"
            );
        }

        return weights_[out_idx * in_features_ + in_idx];
    }

    // Set weights
    void setWeights(const vector<double>& weights) {
        if (weights.size() != in_features_ * out_features_) {
            throw invalid_argument(
                "FullyConnected::setWeights size mismatch"
            );
        }

        for (double value : weights) {
            if (!isfinite(value)) {
                throw invalid_argument(
                    "FullyConnected::setWeights contains NaN or Inf"
                );
            }
        }

        weights_ = weights;
    }

    // Set biases
    void setBias(const vector<double>& biases) {
        if (biases.size() != out_features_) {
            throw invalid_argument(
                "FullyConnected::setBias size mismatch"
            );
        }

        for (double value : biases) {
            if (!isfinite(value)) {
                throw invalid_argument(
                    "FullyConnected::setBias contains NaN or Inf"
                );
            }
        }

        bias_ = biases;
    }

    // Cache management
    bool hasCachedInput() const {
        return has_cache_;
    }

    const ::Tensor& cachedInput() const {
        if (!has_cache_) {
            throw logic_error(
                "FullyConnected::cachedInput called but no cache exists"
            );
        }

        return cached_input_;
    }

    void clearCache() {
        has_cache_ = false;
        cached_input_ = ::Tensor();
    }
};

} // namespace cnn

#endif
