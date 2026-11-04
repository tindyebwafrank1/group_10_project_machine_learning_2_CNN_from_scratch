
#include "cross_entropy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cnn
{

namespace
{

void validateVectors(
    const std::vector<double>& probabilities,
    const std::vector<double>& targets,
    double epsilon
)
{
    if (probabilities.empty() || targets.empty())
    {
        throw std::invalid_argument(
            "Probability and target vectors cannot be empty."
        );
    }

    if (probabilities.size() != targets.size())
    {
        throw std::invalid_argument(
            "Probability and target vectors must have equal sizes."
        );
    }

    if (!std::isfinite(epsilon) || epsilon <= 0.0 || epsilon >= 1.0)
    {
        throw std::invalid_argument(
            "Epsilon must be greater than 0 and less than 1."
        );
    }

    double probabilitySum = 0.0;
    double targetSum = 0.0;

    for (std::size_t i = 0; i < probabilities.size(); ++i)
    {
        double p = probabilities[i];
        double t = targets[i];

        if (!std::isfinite(p) || p < 0.0 || p > 1.0)
        {
            throw std::invalid_argument(
                "Probabilities must be finite values between 0 and 1."
            );
        }

        if (!std::isfinite(t) || t < 0.0)
        {
            throw std::invalid_argument(
                "Targets must be finite and non-negative."
            );
        }

        probabilitySum += p;
        targetSum += t;
    }

    const double tolerance = 1e-6;

    if (std::abs(probabilitySum - 1.0) > tolerance)
    {
        throw std::invalid_argument(
            "Probabilities must sum to 1."
        );
    }

    if (std::abs(targetSum - 1.0) > tolerance)
    {
        throw std::invalid_argument(
            "Targets must sum to 1."
        );
    }
}

} // anonymous namespace


double crossEntropyLoss(
    const std::vector<double>& probabilities,
    const std::vector<double>& targets,
    double epsilon
)
{
    validateVectors(probabilities, targets, epsilon);

    double loss = 0.0;

    for (std::size_t i = 0; i < probabilities.size(); ++i)
    {
        if (targets[i] > 0.0)
        {
            double p = std::max(probabilities[i], epsilon);

            loss -= targets[i] * std::log(p);
        }
    }

    return loss;
}


double crossEntropyLossBatch(
    const std::vector<std::vector<double>>& batchProbabilities,
    const std::vector<std::vector<double>>& batchTargets,
    double epsilon
)
{
    if (batchProbabilities.empty() || batchTargets.empty())
    {
        throw std::invalid_argument(
            "Batch vectors cannot be empty."
        );
    }

    if (batchProbabilities.size() != batchTargets.size())
    {
        throw std::invalid_argument(
            "Probability and target batches must have equal sizes."
        );
    }

    double totalLoss = 0.0;

    for (std::size_t i = 0; i < batchProbabilities.size(); ++i)
    {
        totalLoss += crossEntropyLoss(
            batchProbabilities[i],
            batchTargets[i],
            epsilon
        );
    }

    return totalLoss /
           static_cast<double>(batchProbabilities.size());
}

} // namespace cnn
