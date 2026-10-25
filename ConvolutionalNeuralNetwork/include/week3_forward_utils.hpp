#ifndef WEEK3_FORWARD_UTILS_HPP
#define WEEK3_FORWARD_UTILS_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>
#include "tensor.hpp"

using namespace std;

namespace week3_app
{
    // Flatten all tensor values in their existing storage order.
    inline vector<double> flattenTensor(const Tensor& input)
    {
        if (input.shape.size() != 4 || input.data.empty())
        {
            throw invalid_argument(
                "Flattening requires a non-empty 4D tensor.");
        }

        return input.data;
    }

    // weights[class][feature], with one bias per class.
    inline vector<double> fullyConnected(
        const vector<double>& features,
        const vector<vector<double>>& weights,
        const vector<double>& biases)
    {
        if (features.empty() || weights.empty() ||
            weights.size() != biases.size())
        {
            throw invalid_argument(
                "Invalid fully connected layer dimensions.");
        }

        vector<double> output(weights.size());

        for (size_t c = 0; c < weights.size(); c++)
        {
            if (weights[c].size() != features.size())
            {
                throw invalid_argument(
                    "Weight count must match feature count.");
            }

            output[c] = biases[c];

            for (size_t f = 0; f < features.size(); f++)
            {
                output[c] += weights[c][f] * features[f];
            }
        }

        return output;
    }

    // Numerically stable softmax.
    inline vector<double> softmax(const vector<double>& logits)
    {
        if (logits.empty())
        {
            throw invalid_argument(
                "Softmax requires at least one logit.");
        }

        double maximum = -numeric_limits<double>::infinity();

        for (double value : logits)
        {
            if (!isfinite(value))
            {
                throw invalid_argument(
                    "Softmax input must contain finite values.");
            }

            if (value > maximum)
            {
                maximum = value;
            }
        }

        vector<double> probabilities(logits.size());
        double total = 0.0;

        for (size_t i = 0; i < logits.size(); i++)
        {
            probabilities[i] = exp(logits[i] - maximum);
            total += probabilities[i];
        }

        if (!isfinite(total) || total <= 0.0)
        {
            throw runtime_error(
                "Could not normalize softmax probabilities.");
        }

        for (double& p : probabilities)
        {
            p /= total;
        }

        return probabilities;
    }

    // Return the index of the largest value.
    inline size_t argmax(const vector<double>& values)
    {
        if (values.empty())
        {
            throw invalid_argument(
                "Argmax requires at least one value.");
        }

        size_t best = 0;

        for (size_t i = 0; i < values.size(); i++)
        {
            if (!isfinite(values[i]))
            {
                throw invalid_argument(
                    "Argmax values must be finite.");
            }

            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }

    inline void printShape(const Tensor& tensor)
    {
        cout << "(";

        for (size_t i = 0; i < tensor.shape.size(); i++)
        {
            cout << tensor.shape[i];

            if (i + 1 < tensor.shape.size())
            {
                cout << ", ";
            }
        }

        cout << ")\n";
    }

    inline void printMatrix(const Tensor& tensor)
    {
        if (tensor.shape.size() != 4 ||
            tensor.shape[0] != 1 ||
            tensor.shape[1] != 1)
        {
            throw invalid_argument(
                "Matrix display expects a single-channel 4D tensor.");
        }

        for (size_t h = 0; h < tensor.shape[2]; h++)
        {
            for (size_t w = 0; w < tensor.shape[3]; w++)
            {
                cout << tensor(0, 0, h, w) << "\t";
            }

            cout << "\n";
        }
    }
}

#endif