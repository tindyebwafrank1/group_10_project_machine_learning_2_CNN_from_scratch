
#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

using namespace std;

namespace act {

// Validate tensor data and shape before performing operations.
inline void validateTensor(const Tensor& input)
{
    if (input.shape.size() != 4)
    {
        throw invalid_argument("Activation: input must be a 4-D tensor");
    }

    size_t expected = 1;

    for (size_t dimension : input.shape)
    {
        if (dimension != 0 &&
            expected > input.data.max_size() / dimension)
        {
            throw invalid_argument("Activation: tensor dimensions overflow");
        }

        expected *= dimension;
    }

    if (input.data.size() != expected)
    {
        throw invalid_argument("Activation: tensor data size mismatch");
    }
}

// Validate that two tensors have identical shapes and data sizes.
inline void validateSameShape(
    const Tensor& a,
    const Tensor& b)
{
    validateTensor(a);
    validateTensor(b);

    if (a.shape != b.shape)
    {
        throw invalid_argument(
            "Activation backward: tensor shapes must match");
    }
}

// Numerically stable sigmoid helper.
inline double sigmoid(double x)
{
    if (x >= 0.0)
    {
        return 1.0 / (1.0 + exp(-x));
    }

    double e = exp(x);
    return e / (1.0 + e);
}

// Numerically stable tanh helper.
inline double tanh_stable(double x)
{
    return tanh(x);
}

// =====================================================
// ReLU: f(x) = max(0, x)
// Derivative: 1 for x > 0, otherwise 0
// =====================================================
class ReLU
{
public:
    Tensor forward(const Tensor& input) const
    {
        validateTensor(input);

        Tensor output = input;

        for (size_t i = 0; i < input.data.size(); ++i)
        {
            output.data[i] = max(0.0, input.data[i]);
        }

        return output;
    }

    // output is the result of forward().
    // grad_out is the incoming gradient from the next layer.
    Tensor backward(
        const Tensor& output,
        const Tensor& grad_out) const
    {
        validateSameShape(output, grad_out);

        Tensor grad_input = grad_out;

        for (size_t i = 0; i < output.data.size(); ++i)
        {
            grad_input.data[i] =
                (output.data[i] > 0.0)
                ? grad_out.data[i]
                : 0.0;
        }

        return grad_input;
    }
};

// =====================================================
// Sigmoid: f(x) = 1 / (1 + exp(-x))
// Derivative: f(x) * (1 - f(x))
// =====================================================
class Sigmoid
{
public:
    Tensor forward(const Tensor& input) const
    {
        validateTensor(input);

        Tensor output = input;

        for (size_t i = 0; i < input.data.size(); ++i)
        {
            output.data[i] = sigmoid(input.data[i]);
        }

        return output;
    }

    // Uses the sigmoid output to calculate its derivative.
    Tensor backward(
        const Tensor& output,
        const Tensor& grad_out) const
    {
        validateSameShape(output, grad_out);

        Tensor grad_input = grad_out;

        for (size_t i = 0; i < output.data.size(); ++i)
        {
            double y = output.data[i];

            grad_input.data[i] =
                grad_out.data[i] * y * (1.0 - y);
        }

        return grad_input;
    }
};

// =====================================================
// Tanh: f(x) = tanh(x)
// Derivative: 1 - f(x)^2
// =====================================================
class Tanh
{
public:
    Tensor forward(const Tensor& input) const
    {
        validateTensor(input);

        Tensor output = input;

        for (size_t i = 0; i < input.data.size(); ++i)
        {
            output.data[i] = tanh_stable(input.data[i]);
        }

        return output;
    }

    // Uses the tanh output to calculate its derivative.
    Tensor backward(
        const Tensor& output,
        const Tensor& grad_out) const
    {
        validateSameShape(output, grad_out);

        Tensor grad_input = grad_out;

        for (size_t i = 0; i < output.data.size(); ++i)
        {
            double y = output.data[i];

            grad_input.data[i] =
                grad_out.data[i] * (1.0 - y * y);
        }

        return grad_input;
    }
};

} // namespace act

#endif // ACTIVATIONS_HPP
