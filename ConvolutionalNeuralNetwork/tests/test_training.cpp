#include "training.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

int main()
{
    try
    {
        vector<cnn::TrainingSample> samples = {
            {{1.0, 0.0}, 0},
            {{0.9, 0.1}, 0},
            {{0.0, 1.0}, 1},
            {{0.1, 0.9}, 1},
            {{0.8, 0.2}, 0}
        };

        cnn::SoftmaxClassifier model(2, 2);

        double initial_loss = model.calculateLoss(samples);

        cnn::TrainingConfig config;
        config.epochs = 100;
        config.batch_size = 2;
        config.learning_rate = 0.1;

        cnn::TrainingHistory history = model.train(samples, config);

        if (history.epoch_losses.size() != config.epochs)
        {
            throw runtime_error("Incorrect number of recorded epochs.");
        }

        double final_loss = model.calculateLoss(samples);

        if (!isfinite(final_loss) || final_loss >= initial_loss)
        {
            throw runtime_error("Training did not reduce the loss.");
        }

        for (const cnn::TrainingSample& sample : samples)
        {
            if (model.predict(sample.features) != sample.label)
            {
                throw runtime_error("A training sample was misclassified.");
            }
        }

        cout << "All training tests passed!\n";
        cout << "Initial loss: " << initial_loss << "\n";
        cout << "Final loss: " << final_loss << "\n";

        return 0;
    }
    catch (const exception& e)
    {
        cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}