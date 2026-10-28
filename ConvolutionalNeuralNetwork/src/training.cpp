#include "training.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace std;

namespace cnn
{

SoftmaxClassifier::SoftmaxClassifier(
    size_t number_of_features,
    size_t number_of_classes)
    : feature_count(number_of_features)
{
    if (number_of_features == 0 || number_of_classes < 2)
    {
        throw invalid_argument(
            "Classifier needs features and at least two classes.");
    }

    weights.assign(
        number_of_classes,
        vector<double>(number_of_features, 0.0));

    biases.assign(number_of_classes, 0.0);
}

size_t SoftmaxClassifier::numberOfClasses() const
{
    return weights.size();
}

size_t SoftmaxClassifier::numberOfFeatures() const
{
    return feature_count;
}

vector<double> SoftmaxClassifier::predictProbabilities(
    const vector<double>& features) const
{
    if (features.size() != feature_count)
    {
        throw invalid_argument("Incorrect number of input features.");
    }

    vector<double> logits(weights.size(), 0.0);

    for (size_t c = 0; c < weights.size(); ++c)
    {
        logits[c] = biases[c];

        for (size_t f = 0; f < feature_count; ++f)
        {
            logits[c] += weights[c][f] * features[f];
        }
    }

    // Numerically stable softmax.
    double maximum = *max_element(logits.begin(), logits.end());
    double total = 0.0;

    for (double& value : logits)
    {
        value = exp(value - maximum);
        total += value;
    }

    for (double& value : logits)
    {
        value /= total;
    }

    return logits;
}

size_t SoftmaxClassifier::predict(
    const vector<double>& features) const
{
    vector<double> probabilities = predictProbabilities(features);

    return static_cast<size_t>(
        max_element(probabilities.begin(), probabilities.end())
        - probabilities.begin());
}

double SoftmaxClassifier::calculateLoss(
    const vector<TrainingSample>& samples) const
{
    if (samples.empty())
    {
        throw invalid_argument("Training samples cannot be empty.");
    }

    double total_loss = 0.0;

    for (const TrainingSample& sample : samples)
    {
        if (sample.label >= weights.size())
        {
            throw invalid_argument("Training label is out of range.");
        }

        vector<double> probabilities =
            predictProbabilities(sample.features);

        total_loss -= log(max(
            probabilities[sample.label],
            numeric_limits<double>::min()));
    }

    return total_loss / static_cast<double>(samples.size());
}

void SoftmaxClassifier::trainBatch(
    const vector<TrainingSample>& batch,
    double learning_rate)
{
    if (batch.empty())
    {
        throw invalid_argument("Mini-batch cannot be empty.");
    }

    if (!isfinite(learning_rate) || learning_rate <= 0.0)
    {
        throw invalid_argument("Learning rate must be positive.");
    }

    vector<vector<double>> weight_gradients(
        weights.size(), vector<double>(feature_count, 0.0));

    vector<double> bias_gradients(weights.size(), 0.0);

    for (const TrainingSample& sample : batch)
    {
        if (sample.features.size() != feature_count)
        {
            throw invalid_argument("Incorrect number of input features.");
        }

        if (sample.label >= weights.size())
        {
            throw invalid_argument("Training label is out of range.");
        }

        vector<double> probabilities =
            predictProbabilities(sample.features);

        // For softmax with cross-entropy:
        // gradient of each logit = probability - target.
        for (size_t c = 0; c < weights.size(); ++c)
        {
            double logit_gradient = probabilities[c];

            if (c == sample.label)
            {
                logit_gradient -= 1.0;
            }

            bias_gradients[c] += logit_gradient;

            for (size_t f = 0; f < feature_count; ++f)
            {
                weight_gradients[c][f] +=
                    logit_gradient * sample.features[f];
            }
        }
    }

    double batch_size = static_cast<double>(batch.size());

    // SGD update using the average mini-batch gradient.
    for (size_t c = 0; c < weights.size(); ++c)
    {
        biases[c] -= learning_rate * bias_gradients[c] / batch_size;

        for (size_t f = 0; f < feature_count; ++f)
        {
            weights[c][f] -=
                learning_rate * weight_gradients[c][f] / batch_size;
        }
    }
}

TrainingHistory SoftmaxClassifier::train(
    const vector<TrainingSample>& samples,
    const TrainingConfig& config)
{
    if (samples.empty())
    {
        throw invalid_argument("Training dataset cannot be empty.");
    }

    if (config.epochs == 0 || config.batch_size == 0)
    {
        throw invalid_argument(
            "Epoch count and batch size must be greater than zero.");
    }

    if (!isfinite(config.learning_rate) ||
        config.learning_rate <= 0.0)
    {
        throw invalid_argument("Learning rate must be positive.");
    }

    for (const TrainingSample& sample : samples)
    {
        if (sample.features.size() != feature_count)
        {
            throw invalid_argument("Training feature size mismatch.");
        }

        if (sample.label >= weights.size())
        {
            throw invalid_argument("Training label is out of range.");
        }
    }

    TrainingHistory history;

    for (size_t epoch = 0; epoch < config.epochs; ++epoch)
    {
        // Process the dataset in mini-batches.
        for (size_t start = 0; start < samples.size();
             start += config.batch_size)
        {
            size_t end = min(
                start + config.batch_size, 
                samples.size());

            vector<TrainingSample> batch(
                samples.begin() + static_cast<ptrdiff_t>(start),
                samples.begin() + static_cast<ptrdiff_t>(end));

            trainBatch(batch, config.learning_rate);
        }

        // Record loss after completing each epoch.
        history.epoch_losses.push_back(calculateLoss(samples));
    }

    return history;
}

} // namespace cnn