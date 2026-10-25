#include "training.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

int main()
{
    // Features represent outputs from an earlier feature-extraction stage.
    // Class 0 has its first feature active; class 1 has its second active.
    vector<cnn::TrainingSample> samples = {
        {{1.0, 0.0}, 0},
        {{0.9, 0.1}, 0},
        {{0.0, 1.0}, 1},
        {{0.1, 0.9}, 1}
    };

    cnn::SoftmaxClassifier model(2, 2);

    cnn::TrainingConfig config;
    config.epochs = 100;
    config.batch_size = 2;
    config.learning_rate = 0.1;

    cout << "GROUP 10 CNN - WEEK 4 TRAINING DEMO\n";
    cout << "Training examples: " << samples.size() << "\n";
    cout << "Mini-batch size: " << config.batch_size << "\n";
    cout << "Epochs: " << config.epochs << "\n\n";

    double initial_loss = model.calculateLoss(samples);

    cnn::TrainingHistory history = model.train(samples, config);

    cout << fixed << setprecision(6);
    cout << "Initial loss: " << initial_loss << "\n";
    cout << "Final loss:   " << history.epoch_losses.back() << "\n\n";

    cout << "Loss during training:\n";

    for (size_t i = 0; i < history.epoch_losses.size(); i += 10)
    {
        cout << "Epoch " << (i + 1) << ": "
             << history.epoch_losses[i] << "\n";
    }

    cout << "\nPredictions after training:\n";

    for (size_t i = 0; i < samples.size(); ++i)
    {
        size_t prediction = model.predict(samples[i].features);

        cout << "Sample " << (i + 1)
             << " | Expected: " << samples[i].label
             << " | Predicted: " << prediction << "\n";
    }

    return 0;
}