#ifndef CNN_TRAINING_HPP
#define CNN_TRAINING_HPP

#include <cstddef>
#include <vector>

namespace cnn
{

struct TrainingSample
{
    std::vector<double> features;
    std::size_t label;
};

struct TrainingConfig
{
    std::size_t epochs = 100;
    std::size_t batch_size = 4;
    double learning_rate = 0.1;
};

struct TrainingHistory
{
    std::vector<double> epoch_losses;
};

class SoftmaxClassifier
{
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::size_t feature_count;

public:
    SoftmaxClassifier(
        std::size_t number_of_features,
        std::size_t number_of_classes);

    std::vector<double> predictProbabilities(
        const std::vector<double>& features) const;

    std::size_t predict(
        const std::vector<double>& features) const;

    double calculateLoss(
        const std::vector<TrainingSample>& samples) const;

    void trainBatch(
        const std::vector<TrainingSample>& batch,
        double learning_rate);

    TrainingHistory train(
        const std::vector<TrainingSample>& samples,
        const TrainingConfig& config);

    std::size_t numberOfClasses() const;
    std::size_t numberOfFeatures() const;
};

} // namespace cnn

#endif