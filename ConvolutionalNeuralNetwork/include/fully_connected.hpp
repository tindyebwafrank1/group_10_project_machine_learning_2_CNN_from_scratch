#ifndef FULLY_CONNECTED_HPP
#define FULLY_CONNECTED_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <limits>
#include <algorithm>
#include <iostream>
#include "tensor.hpp"

namespace cnn {

class FullyConnected {
public:  
  FullyConnected(size_t in_features, size_t out_features)
        : in_features_(in_features), 
          out_features_(out_features),
          cached_input_(0, 0, 0, 0) // <--- Initialize the Tensor here
    {
        
        if (in_features == 0 || out_features == 0) {
            throw std::invalid_argument("FullyConnected: feature dimensions must be positive and valid");
        }

        // Initialize weights and biases to zero
        weights_.assign(in_features_ * out_features_, 0.0);
        bias_.assign(out_features_, 0.0);
    }

    // Forward pass: computes output and caches input
    Tensor forward(const Tensor& input) {
        validateInput(input);
        
        Tensor output = computeForward(input);
        
        // Cache a copy of the input
        cached_input_ = input;
        has_cache_ = true;
        
        return output;
    }

    // Inference: computes output without touching the cache
    Tensor infer(const Tensor& input) const {
        validateInput(input);
        return computeForward(input);
    }

    // Accessors
    size_t inFeatures() const { return in_features_; }
    size_t outFeatures() const { return out_features_; }
    size_t parameterCount() const { return weights_.size() + bias_.size(); }
    
    const std::vector<double>& weights() const { return weights_; }
    const std::vector<double>& bias() const { return bias_; }

    double weight(size_t out_idx, size_t in_idx) const {
        if (out_idx >= out_features_ || in_idx >= in_features_) {
            throw std::out_of_range("FullyConnected::weight index out of range");
        }
        return weights_[out_idx * in_features_ + in_idx];
    }

    // Setters
    void setWeights(const std::vector<double>& w) {
        if (w.size() != in_features_ * out_features_) {
            throw std::invalid_argument("FullyConnected::setWeights size mismatch");
        }
        for (double val : w) {
            if (std::isnan(val) || std::isinf(val)) {
                throw std::invalid_argument("FullyConnected::setWeights contains NaN or Inf");
            }
        }
        weights_ = w;
    }

    void setBias(const std::vector<double>& b) {
        if (b.size() != out_features_) {
            throw std::invalid_argument("FullyConnected::setBias size mismatch");
        }
        for (double val : b) {
            if (std::isnan(val) || std::isinf(val)) {
                throw std::invalid_argument("FullyConnected::setBias contains NaN or Inf");
            }
        }
        bias_ = b;
    }

    // Cache management
    bool hasCachedInput() const { return has_cache_; }
    
    const Tensor& cachedInput() const {
        if (!has_cache_) {
            throw std::logic_error("FullyConnected::cachedInput called but no cache exists");
        }
        return cached_input_;
    }
    
    void clearCache() {
        has_cache_ = false;
        // Reset to an empty, 0-dimensional tensor using your existing constructor
        cached_input_ = Tensor(0, 0, 0, 0); 
    }

private:
    size_t in_features_;
    size_t out_features_;
    std::vector<double> weights_;
    std::vector<double> bias_;
    
    bool has_cache_ = false;
    Tensor cached_input_;

    void validateInput(const Tensor& input) const {
        if (input.shape.size() != 4) {
            throw std::invalid_argument("FullyConnected: input must be 4-D tensor {N, C, H, W}");
        }
        
        // Calculate total features per sample (C * H * W)
        size_t sample_features = input.shape[1] * input.shape[2] * input.shape[3];
        if (sample_features != in_features_) {
            throw std::invalid_argument("FullyConnected: input feature dimension mismatch");
        }

        size_t n = input.shape[0];
        if (input.data.size() != n * in_features_) {
            throw std::invalid_argument("FullyConnected: input data size does not match shape");
        }

        // Validate data values
        for (double val : input.data) {
            if (std::isnan(val) || std::isinf(val)) {
                throw std::invalid_argument("FullyConnected: input contains NaN or Inf");
            }
        }
    }

    Tensor computeForward(const Tensor& input) const {
        size_t N = input.shape[0];
        Tensor output(N, out_features_, 1, 1);

        for (size_t n = 0; n < N; ++n) {
            for (size_t o = 0; o < out_features_; ++o) {
                double sum = bias_[o];
                for (size_t i = 0; i < in_features_; ++i) {
                    double w = weights_[o * in_features_ + i];
                    double x = input.data[n * in_features_ + i];
                    
                    // Check for overflow before multiplication
                    if (std::abs(x) > 1.0 && std::abs(w) > std::numeric_limits<double>::max() / std::abs(x)) {
                        throw std::runtime_error("FullyConnected: overflow detected during multiplication");
                    }
                    
                    double prod = w * x;
                    // Check for overflow during addition
                    if (std::abs(sum) > std::numeric_limits<double>::max() - std::abs(prod) && 
                        (sum > 0) == (prod > 0)) {
                        throw std::runtime_error("FullyConnected: overflow detected during accumulation");
                    }
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