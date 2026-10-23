#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP
// All layers are applied element-wise: every value in the tensor
// is passed through the activation function independently, so the
// output has exactly the same shape as the input.

#include <cmath> // std::exp, std::tanh
#include <algorithm> // std::max
#include <stdexcept> // std::invalid_argument
#include <cstddef> // size_t
#include "tensor.hpp" // Tensor: public data (flat vector) and shape {N,C,H,W}

namespace act {

// Numerically stable sigmoid: sigma(x) = 1 / (1 + e^-x)
// A direct formula overflows for large negative x, because e^-x
// becomes huge (e.g. e^1000 = inf). So we use two equivalent forms:
// x >= 0: 1 / (1 + e^-x) -> e^-x is in (0, 1], safe
// x < 0: e^x / (1 + e^x) -> e^x is in (0, 1), safe
// exp() is therefore never called on a large positive number.
inline double sigmoid(double x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    const double e = std::exp(x);
    return e / (1.0 + e);
}

// Stable tanh: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
// The textbook formula gives inf/inf = NaN for large |x|.
// std::tanh handles this internally and saturates to +1 or -1.
inline double tanh_stable(double x) {
    return std::tanh(x);
}

// ReLU layer: ReLU(x) = max(0, x), negative values become 0.
class ReLU {
public:
    // Forward pass: returns a new tensor, the input is not modified.
    Tensor forward(const Tensor& input) const {
        Tensor out = input; // copy shape and data
        for (double& v : out.data) {
            v = std::max(0.0, v); // apply to each element
        }
        return out;
    }

    // Backward pass (needed in Week 4).
    // Derivative: ReLU'(x) = 1 if x > 0, otherwise 0 (we use 0 at x = 0).
    // y > 0 exactly when x > 0, so the saved output y can be used,
    // the same way as Sigmoid and Tanh.
    Tensor backward(const Tensor& output, const Tensor& grad_out) const {
        if (output.data.size() != grad_out.data.size()) {
            throw std::invalid_argument("ReLU::backward: size mismatch");
        }
        Tensor g = grad_out;
        for (size_t i = 0; i < g.data.size(); ++i) {
            if (output.data[i] <= 0.0) {
                g.data[i] = 0.0; // dL/dx = 0 where ReLU was inactive
            } // else dL/dx = dL/dy * 1
        }
        return g;
    }
};

// Sigmoid layer: squashes every value into the range (0, 1).
class Sigmoid {
public:
    // Forward pass: returns a new tensor, the input is not modified.
    Tensor forward(const Tensor& input) const {
        Tensor out = input; // copy shape and data
        for (double& v : out.data) {
            v = sigmoid(v); // apply to each element
        }
        return out;
    }

    // Backward pass (needed in Week 4).
    // Derivative: sigma'(x) = y * (1 - y), where y = sigma(x).
    // 'output' is the saved forward result y, 'grad_out' is dL/dy
    // coming from the next layer. Returns dL/dx by the chain rule.
    Tensor backward(const Tensor& output, const Tensor& grad_out) const {
        if (output.data.size() != grad_out.data.size()) {
            throw std::invalid_argument("Sigmoid::backward: size mismatch");
        }
        Tensor g = grad_out;
        for (size_t i = 0; i < g.data.size(); ++i) {
            const double y = output.data[i];
            g.data[i] *= y * (1.0 - y); // dL/dx = dL/dy * sigma'(x)
        }
        return g;
    }
};

// Tanh layer: squashes every value into the range (-1, 1).
class Tanh {
public:
    // Forward pass: returns a new tensor, the input is not modified.
    Tensor forward(const Tensor& input) const {
        Tensor out = input; // copy shape and data
        for (double& v : out.data) {
            v = tanh_stable(v); // apply to each element
        }
        return out;
    }

    // Backward pass (needed in Week 4).
    // Derivative: tanh'(x) = 1 - y^2, where y = tanh(x).
    Tensor backward(const Tensor& output, const Tensor& grad_out) const {
        if (output.data.size() != grad_out.data.size()) {
            throw std::invalid_argument("Tanh::backward: size mismatch");
        }
        Tensor g = grad_out;
        for (size_t i = 0; i < g.data.size(); ++i) {
            const double y = output.data[i];
            g.data[i] *= 1.0 - y * y; // dL/dx = dL/dy * tanh'(x)
        }
        return g;
    }
};

} // namespace act

#endif // ACTIVATIONS_HPP
