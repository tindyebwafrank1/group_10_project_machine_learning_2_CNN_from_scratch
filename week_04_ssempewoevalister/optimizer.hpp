#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

// Week 4 - Evalister: Stochastic Gradient Descent optimizer.
//
//   plain SGD :  W <- W - lr * grad
//   with extras (both optional, off by default):
//     weight decay (L2):  grad_eff = grad + weight_decay * W
//     momentum:           v <- momentum * v - lr * grad_eff ;  W <- W + v
//
// Header-only, namespace CNN. Depends only on tensor.hpp Agatha.

#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnn_library/tensor.hpp"

namespace CNN {

    class SGD {
    private:
        double lr;
        double momentum_;
        double weight_decay_;
        // Momentum state, one velocity vector per parameter, keyed by the parameter's address.
        std::map<const void*, std::vector<double>> velocity;

        static void check_lr(double v) {
            if (!(v > 0.0) || !std::isfinite(v))
                throw std::invalid_argument("SGD: learning rate must be a positive finite number");
        }

    public:
        explicit SGD(double learning_rate, double momentum = 0.0, double weight_decay = 0.0)
            : lr(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {
            check_lr(lr);
            if (!(momentum_ >= 0.0 && momentum_ < 1.0))
                throw std::invalid_argument("SGD: momentum must be in [0, 1)");
            if (!(weight_decay_ >= 0.0) || !std::isfinite(weight_decay_))
                throw std::invalid_argument("SGD: weight decay must be >= 0 and finite");
        }

        double learning_rate() const { return lr; }
        double momentum() const { return momentum_; }
        double weight_decay() const { return weight_decay_; }

        void set_learning_rate(double v) { check_lr(v); lr = v; }

        // Multiply the learning rate by `factor` in (0, 1], e.g. once per epoch.
        void decay_learning_rate(double factor) {
            if (!(factor > 0.0 && factor <= 1.0))
                throw std::invalid_argument("SGD: decay factor must be in (0, 1]");
            lr *= factor;
        }

        // Forget all momentum state (e.g. when starting a new training run).
        void reset_state() { velocity.clear(); }

        // Update one parameter array in place.
        // grad_scale multiplies the gradient first: pass 1.0 if gradients are already
        // batch-averaged, or 1.0 / batch_size if they were summed over the batch.
        void step(std::vector<double>& params, const std::vector<double>& grads, double grad_scale = 1.0) {
            if (params.size() != grads.size())
                throw std::invalid_argument("SGD: params and grads must have the same size (" +
                                            std::to_string(params.size()) + " vs " +
                                            std::to_string(grads.size()) + ")");
            // Validate everything before touching params, so a bad gradient never half-updates.
            for (double g : grads)
                if (!std::isfinite(g * grad_scale))
                    throw std::runtime_error("SGD: non-finite gradient (training has diverged or a backward pass is wrong)");

            const size_t n = params.size();
            if (momentum_ == 0.0) {
                for (size_t i = 0; i < n; ++i)
                    params[i] -= lr * (grads[i] * grad_scale + weight_decay_ * params[i]);
                return;
            }

            std::vector<double>& v = velocity[&params];
            if (v.empty()) v.assign(n, 0.0);
            else if (v.size() != n)
                throw std::invalid_argument("SGD: parameter size changed since the last step; call reset_state()");
            for (size_t i = 0; i < n; ++i) {
                double g = grads[i] * grad_scale + weight_decay_ * params[i];
                v[i] = momentum_ * v[i] - lr * g;
                params[i] += v[i];
            }
        }

        // Tensor overload: shapes must match exactly.
        void step(Tensor& param, const Tensor& grad, double grad_scale = 1.0) {
            if (param.shape != grad.shape)
                throw std::invalid_argument("SGD: param and grad tensors must have the same shape");
            step(param.data, grad.data, grad_scale);
        }
    };

    // Set every element to zero, to clear accumulated gradients between mini-batches.
    inline void zero_gradients(std::vector<double>& grads) { grads.assign(grads.size(), 0.0); }
    inline void zero_gradients(Tensor& grads) { zero_gradients(grads.data); }
}

#endif // OPTIMIZER_HPP
