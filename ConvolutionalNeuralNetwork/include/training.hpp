
#ifndef CNN_TRAINING_HPP
#define CNN_TRAINING_HPP

#include <cstddef>
#include <vector>

using namespace std;

namespace cnn
{

struct TrainingSample
{
    vector<double> features;
    size_t label;
};

struct TrainingConfig
{
    size_t epochs = 100;
    size_t batch_size = 4;
    double learning_rate = 0.1;
};

struct TrainingHistory
{
    vector<double> epoch_losses;
};

class SoftmaxClassifier
{
private:
    vector<vector<double>> weights;
    vector<double> biases;
    size_t feature_count;

public:
    SoftmaxClassifier(
        size_t number_of_features,
        size_t number_of_classes);

    vector<double> predictProbabilities(
        const vector<double>& features) const;

    size_t predict(
        const vector<double>& features) const;

    double calculateLoss(
        const vector<TrainingSample>& samples) const;

    void trainBatch(
        const vector<TrainingSample>& batch,
        double learning_rate);

    TrainingHistory train(
        const vector<TrainingSample>& samples,
        const TrainingConfig& config);

    size_t numberOfClasses() const;
    size_t numberOfFeatures() const;
};

} // namespace cnn

#endif
