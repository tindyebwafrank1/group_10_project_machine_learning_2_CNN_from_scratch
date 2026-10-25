#ifndef FULLY_CONNECTED_HPP
#define FULLY_CONNECTED_HPP


#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <limits>
#include <algorithm>

#include "tensor.hpp"  

namespace cnn {

class FullyConnected {
public:
    FullyConnected(size_t in_features, size_t out_features)
        : in_features_(in_features), out_features_(out_features) {

        if (in_features == 0 || out_features == 0 ||
            in_features == static_cast<size_t>(-1) || out_features == static_cast<size_t>(-1)) {
            throw std::invalid_argument("FullyConnected: feature dimensions must be positive and valid");
        }

        weights_.assign(in_features_ * out_features_, 0.0);
        bias_.assign(out_features_, 0.0);
    }

    //forward 
    ::Tensor forward(const ::Tensor& input) {
        validateInput(input);
        ::Tensor output = computeForward(input);
        cached_input_ = input;   // copy, not reference
        has_cache_ = true;
        return output;
    }

    ::Tensor infer(const ::Tensor& input) const {
        validateInput(input);
        return computeForward(input);
    }

    // accessors
    size_t inFeatures()     const { return in_features_; }
    size_t outFeatures()    const { return out_features_; }
    size_t parameterCount() const { return weights_.size() + bias_.size(); }

    const std::vector<double>& weights() const { return weights_; }
    const std::vector<double>& bias()    const { return bias_; }

    double weight(size_t out_idx, size_t in_idx) const {
        if (out_idx >= out_features_ || in_idx >= in_features_) {
            throw std::out_of_range("FullyConnected::weight index out of range");
        }
        return weights_[out_idx * in_features_ + in_idx];
    }

    // setters 
    void setWeights(const std::vector<double>& w) {
        if (w.size() != in_features_ * out_features_) {
            throw std::invalid_argument("FullyConnected::setWeights size mismatch");
        }
        for (double v : w)
            if (std::isnan(v) || std::isinf(v))
                throw std::invalid_argument("FullyConnected::setWeights contains NaN or Inf");
        weights_ = w;
    }

    void setBias(const std::vector<double>& b) {
        if (b.size() != out_features_) {
            throw std::invalid_argument("FullyConnected::setBias size mismatch");
        }
        for (double v : b)
            if (std::isnan(v) || std::isinf(v))
                throw std::invalid_argument("FullyConnected::setBias contains NaN or Inf");
        bias_ = b;
    }

    //cache-
    bool hasCachedInput() const { return has_cache_; }

    const ::Tensor& cachedInput() const {
        if (!has_cache_)
            throw std::logic_error("FullyConnected::cachedInput called but no cache exists");
        return cached_input_;
    }

    void clearCache() {
        has_cache_ = false;
        cached_input_ = ::Tensor();
    }

private:
    size_t in_features_;
    size_t out_features_;
    std::vector<double> weights_;
    std::vector<double> bias_;

    bool has_cache_ = false;
    ::Tensor cached_input_;

    //helpers
    void validateInput(const ::Tensor& input) const {
        if (input.shape.size() != 4)
            throw std::invalid_argument("FullyConnected: input must be a 4-D tensor {N, C, H, W}");

        size_t sample_features = input.shape[1] * input.shape[2] * input.shape[3];
        if (sample_features != in_features_)
            throw std::invalid_argument(
                "FullyConnected: expected " + std::to_string(in_features_) +
                " features per sample, but got " + std::to_string(sample_features));

        size_t n = input.shape[0];
        if (input.data.size() != n * in_features_)
            throw std::invalid_argument("FullyConnected: input data size does not match shape");

        for (double v : input.data)
            if (std::isnan(v) || std::isinf(v))
                throw std::invalid_argument("FullyConnected: input contains NaN or Inf");
    }

    ::Tensor computeForward(const ::Tensor& input) const {
        size_t N = input.shape[0];
        ::Tensor output(N, out_features_, 1, 1);

        for (size_t n = 0; n < N; ++n) {
            for (size_t o = 0; o < out_features_; ++o) {
                double sum = bias_[o];
                for (size_t i = 0; i < in_features_; ++i) {
                    double w = weights_[o * in_features_ + i];
                    double x = input.data[n * in_features_ + i];

                    if (std::abs(x) > 1.0 &&
                        std::abs(w) > std::numeric_limits<double>::max() / std::abs(x))
                        throw std::runtime_error("FullyConnected: overflow detected during multiplication");

                    double prod = w * x;
                    if (std::abs(sum) > std::numeric_limits<double>::max() - std::abs(prod) &&
                        (sum > 0) == (prod > 0))
                        throw std::runtime_error("FullyConnected: overflow detected during accumulation");

                    sum += prod;
                }
                output.data[n * out_features_ + o] = sum;
            }
        }
        return output;
    }
};

} // namespace cnn

#endif // FULLY_CONNECTED_HPP