#include "cross_entropy.hpp"

#include <cmath>
#include <stdexcept>

using namespace std;

// Validates probability and target vectors.
static void validateVectors(
    const vector<double>& probabilities,
    const vector<double>& targets,
    double epsilon)
{
    if (probabilities.empty() ||
        probabilities.size() != targets.size())
    {
        throw invalid_argument(
            "Probabilities and targets must have the same non-zero size."
        );
    }

    if (!isfinite(epsilon) || epsilon <= 0.0 || epsilon >= 1.0)
    {
        throw invalid_argument(
            "Epsilon must be greater than zero and less than one."
        );
    }

    double probability_sum = 0.0;
    double target_sum = 0.0;

    for (size_t i = 0; i < probabilities.size(); i++)
    {
        double p = probabilities[i];
        double y = targets[i];

        if (!isfinite(p) || p < 0.0 || p > 1.0)
        {
            throw invalid_argument(
                "Probabilities must be finite values between zero and one."
            );
        }

        if (!isfinite(y) || y < 0.0)
        {
            throw invalid_argument(
                "Targets must be finite and non-negative."
            );
        }

        probability_sum += p;
        target_sum += y;
    }

    // Permit small floating-point rounding differences.
    const double tolerance = 1e-6;

    if (abs(probability_sum - 1.0) > tolerance)
    {
        throw invalid_argument(
            "Predicted probabilities must sum to one."
        );
    }

    if (abs(target_sum - 1.0) > tolerance)
    {
        throw invalid_argument(
            "Target values must sum to one."
        );
    }
}

double crossEntropyLoss(
    const vector<double>& probabilities,
    const vector<double>& targets,
    double epsilon)
{
    validateVectors(probabilities, targets, epsilon);

    double loss = 0.0;

    for (size_t i = 0; i < probabilities.size(); i++)
    {
        // Classes with target zero contribute nothing.
        if (targets[i] == 0.0)
        {
            continue;
        }

        // Prevent log(0) if a predicted probability is zero.
        double p = probabilities[i];

        if (p < epsilon)
        {
            p = epsilon;
        }

        loss -= targets[i] * log(p);
    }

    return loss;
}

double crossEntropyLossBatch(
    const vector<vector<double>>& probabilities,
    const vector<vector<double>>& targets,
    double epsilon)
{
    if (probabilities.empty() ||
        probabilities.size() != targets.size())
    {
        throw invalid_argument(
            "Probability and target batches must have the same non-zero size."
        );
    }

    double total_loss = 0.0;

    for (size_t i = 0; i < probabilities.size(); i++)
    {
        total_loss += crossEntropyLoss(
            probabilities[i],
            targets[i],
            epsilon
        );
    }

    // Return mean loss across the batch.
    return total_loss / static_cast<double>(probabilities.size());
}
