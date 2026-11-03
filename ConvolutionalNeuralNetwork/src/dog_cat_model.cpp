
#include "../include/dog_cat_model.hpp"
#include "../include/preprocess.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <random>
#include <numeric>

using namespace std;

namespace cnn {

const size_t IMAGE_HEIGHT = 32;
const size_t IMAGE_WIDTH = 32;
const size_t BATCH_SIZE = 8;

DogCatModel::DogCatModel()
{
    feature_count = IMAGE_HEIGHT * IMAGE_WIDTH;
}

bool DogCatModel::extractFeatures(
    const string& filename,
    vector<double>& features
) const
{
    Image image;

    if (!loadImage(filename, image)) {
        cerr << "Could not load image: " << filename << '\n';
        return false;
    }

    int width = image.getWidth();
    int height = image.getHeight();
    int channels = image.getChannels();

    if (width <= 0 || height <= 0 || channels != 3) {
        cerr << "Invalid RGB image: " << filename << '\n';
        return false;
    }

    const vector<Pixel>& pixels = image.getPixels();

    if (pixels.size() !=
        static_cast<size_t>(width) * static_cast<size_t>(height)) {
        cerr << "Invalid image pixel data: " << filename << '\n';
        return false;
    }

    vector<unsigned char> rgb;
    rgb.reserve(pixels.size() * 3);

    for (const Pixel& pixel : pixels) {
        rgb.push_back(pixel.r);
        rgb.push_back(pixel.g);
        rgb.push_back(pixel.b);
    }

    Tensor input = cnn::fromInterleaved(
        rgb.data(),
        rgb.size(),
        static_cast<size_t>(height),
        static_cast<size_t>(width),
        3
    );

    cnn::PreprocessConfig config;

    config.target_channels = 1;
    config.target_height = IMAGE_HEIGHT;
    config.target_width = IMAGE_WIDTH;
    config.interpolation = cnn::Interpolation::Bilinear;
    config.normalization = cnn::Normalization::Scale;

    Tensor processed = cnn::preprocess(input, config);

    if (processed.data.size() != IMAGE_HEIGHT * IMAGE_WIDTH) {
        cerr << "Unexpected processed image size: "
             << filename << '\n';
        return false;
    }

    features = processed.data;

    return true;
}

void DogCatModel::train(
    const vector<DogCatSample>& samples,
    size_t epochs,
    double learning_rate
)
{
    if (samples.empty()) {
        throw invalid_argument("Training dataset is empty.");
    }

    if (epochs == 0 ||
        !isfinite(learning_rate) ||
        learning_rate <= 0.0) {
        throw invalid_argument("Invalid training configuration.");
    }

    feature_count = samples[0].features.size();

    if (feature_count == 0) {
        throw invalid_argument("Feature vector is empty.");
    }

    // Validate all training samples.
    for (const DogCatSample& sample : samples) {

        if (sample.features.size() != feature_count) {
            throw invalid_argument("Inconsistent feature sizes.");
        }

        if (sample.label > 1) {
            throw invalid_argument("Labels must be 0 or 1.");
        }

        for (double value : sample.features) {
            if (!isfinite(value)) {
                throw invalid_argument(
                    "Training features contain invalid values."
                );
            }
        }
    }

    // Initialize weights with small random values.
    mt19937 generator(42);

    double limit = sqrt(6.0 / (feature_count + 2));

    uniform_real_distribution<double> distribution(
        -limit,
        limit
    );

    weights.assign(
        2,
        vector<double>(feature_count)
    );

    biases.assign(2, 0.0);

    for (size_t c = 0; c < 2; ++c) {
        for (size_t j = 0; j < feature_count; ++j) {
            weights[c][j] = distribution(generator);
        }
    }

    // Training sample order.
    vector<size_t> indices(samples.size());

    iota(indices.begin(), indices.end(), 0);

    // Shuffle generator.
    mt19937 shuffleGenerator(12345);

    vector<double> gradientWeights(
        2 * feature_count,
        0.0
    );

    vector<double> gradientBiases(2, 0.0);

    cout << "Training using mini-batch gradient descent..."
         << '\n';

    for (size_t epoch = 0; epoch < epochs; ++epoch) {

        shuffle(
            indices.begin(),
            indices.end(),
            shuffleGenerator
        );

        double total_loss = 0.0;

        for (size_t start = 0;
             start < samples.size();
             start += BATCH_SIZE) {

            fill(
                gradientWeights.begin(),
                gradientWeights.end(),
                0.0
            );

            fill(
                gradientBiases.begin(),
                gradientBiases.end(),
                0.0
            );

            size_t end = min(
                start + BATCH_SIZE,
                samples.size()
            );

            size_t currentBatchSize = end - start;

            // Calculate gradients for this mini-batch.
            for (size_t i = start; i < end; ++i) {

                const DogCatSample& sample =
                    samples[indices[i]];

                vector<double> probabilities =
                    predictProbabilities(sample.features);

                total_loss -= log(
                    max(
                        probabilities[sample.label],
                        1e-12
                    )
                );

                for (size_t c = 0; c < 2; ++c) {

                    double gradient = probabilities[c];

                    if (c == sample.label) {
                        gradient -= 1.0;
                    }

                    gradientBiases[c] += gradient;

                    for (size_t j = 0;
                         j < feature_count;
                         ++j) {

                        gradientWeights[
                            c * feature_count + j
                        ] += gradient * sample.features[j];
                    }
                }
            }

            // Update weights and biases.
            double scale =
                learning_rate / currentBatchSize;

            for (size_t c = 0; c < 2; ++c) {

                biases[c] -= scale * gradientBiases[c];

                for (size_t j = 0;
                     j < feature_count;
                     ++j) {

                    weights[c][j] -= scale *
                        gradientWeights[
                            c * feature_count + j
                        ];
                }
            }
        }

        if ((epoch + 1) % 10 == 0 || epoch == 0) {

            cout << "Epoch " << epoch + 1
                 << " / " << epochs
                 << " | Loss: "
                 << total_loss / samples.size()
                 << '\n';
        }
    }
}

vector<double> DogCatModel::predictProbabilities(
    const vector<double>& features
) const
{
    if (weights.size() != 2 ||
        biases.size() != 2 ||
        features.size() != feature_count) {

        throw invalid_argument(
            "Model or feature size is invalid."
        );
    }

    vector<double> logits(2, 0.0);

    // Calculate class scores.
    for (size_t c = 0; c < 2; ++c) {

        logits[c] = biases[c];

        for (size_t j = 0; j < feature_count; ++j) {
            logits[c] += weights[c][j] * features[j];
        }
    }

    // Numerically stable softmax.
    double maximum = max(logits[0], logits[1]);

    double sum = 0.0;

    for (double& value : logits) {

        value = exp(value - maximum);

        sum += value;
    }

    if (sum <= 0.0 || !isfinite(sum)) {
        throw runtime_error("Invalid softmax calculation.");
    }

    for (double& value : logits) {
        value /= sum;
    }

    return logits;
}

size_t DogCatModel::predict(
    const vector<double>& features
) const
{
    vector<double> probabilities =
        predictProbabilities(features);

    return probabilities[1] > probabilities[0] ? 1 : 0;
}

bool DogCatModel::save(const string& filename) const
{
    if (weights.size() != 2 ||
        biases.size() != 2 ||
        weights[0].size() != feature_count ||
        weights[1].size() != feature_count) {

        return false;
    }

    ofstream file(filename);

    if (!file.is_open()) {
        return false;
    }

    file.precision(numeric_limits<double>::max_digits10);

    file << feature_count << '\n';

    for (size_t c = 0; c < 2; ++c) {

        file << biases[c] << '\n';

        for (double weight : weights[c]) {
            file << weight << ' ';
        }

        file << '\n';
    }

    return file.good();
}

bool DogCatModel::load(const string& filename)
{
    ifstream file(filename);

    if (!file.is_open()) {
        return false;
    }

    size_t loadedFeatures;

    if (!(file >> loadedFeatures) ||
        loadedFeatures == 0) {

        return false;
    }

    vector<vector<double>> loadedWeights(
        2,
        vector<double>(loadedFeatures)
    );

    vector<double> loadedBiases(2);

    for (size_t c = 0; c < 2; ++c) {

        if (!(file >> loadedBiases[c]) ||
            !isfinite(loadedBiases[c])) {

            return false;
        }

        for (size_t j = 0; j < loadedFeatures; ++j) {

            if (!(file >> loadedWeights[c][j]) ||
                !isfinite(loadedWeights[c][j])) {

                return false;
            }
        }
    }

    feature_count = loadedFeatures;
    weights = loadedWeights;
    biases = loadedBiases;

    return true;
}

size_t DogCatModel::getFeatureCount() const
{
    return feature_count;
}

}
