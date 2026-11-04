#include "../include/dog_cat_model.hpp"
#include "../include/preprocess.hpp"
#include "../include/multi_channel_convolution.hpp"
#include "../include/convolution_backward.hpp"
#include "../include/relu.hpp"
#include "../include/max_pooling.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <random>
#include <numeric>

namespace cnn {

namespace {

const size_t IMAGE_HEIGHT = 32;
const size_t IMAGE_WIDTH = 32;
const size_t INPUT_CHANNELS = 3;

const size_t FILTER_COUNT = 8;
const size_t KERNEL_SIZE = 3;

const size_t CONV_HEIGHT =
    IMAGE_HEIGHT - KERNEL_SIZE + 1;

const size_t CONV_WIDTH =
    IMAGE_WIDTH - KERNEL_SIZE + 1;

const size_t POOL_HEIGHT = CONV_HEIGHT / 2;
const size_t POOL_WIDTH = CONV_WIDTH / 2;

const size_t DENSE_INPUT =
    FILTER_COUNT * POOL_HEIGHT * POOL_WIDTH;

const size_t CLASS_COUNT = 2;

const double WEIGHT_DECAY = 0.00001;

struct ForwardCache {
    Tensor input;
    Tensor conv_pre_activation;
    Tensor relu_output;
    Tensor pool_output;

    std::vector<size_t> pool_indices;
    std::vector<double> probabilities;
};

Tensor createInputTensor(
    const std::vector<double>& features
) {
    const size_t expected =
        INPUT_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH;

    if (features.size() != expected) {
        throw std::invalid_argument(
            "Incorrect image feature size."
        );
    }

    Tensor input(
        1,
        INPUT_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH
    );

    input.data = features;

    return input;
}

std::vector<double> softmax(
    const std::vector<double>& logits
) {
    if (logits.empty()) {
        throw std::invalid_argument(
            "Softmax input cannot be empty."
        );
    }

    double maximum = *std::max_element(
        logits.begin(),
        logits.end()
    );

    std::vector<double> probabilities(
        logits.size(),
        0.0
    );

    double sum = 0.0;

    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(
            logits[i] - maximum
        );

        sum += probabilities[i];
    }

    if (!std::isfinite(sum) || sum <= 0.0) {
        throw std::runtime_error(
            "Invalid softmax calculation."
        );
    }

    for (double& value : probabilities) {
        value /= sum;
    }

    return probabilities;
}

ForwardCache forwardPass(
    const Tensor& input,
    const Tensor& filters,
    const std::vector<double>& convBiases,
    const std::vector<std::vector<double>>& denseWeights,
    const std::vector<double>& denseBiases
) {
    ForwardCache cache;

    cache.input = input;

    // 1. Multi-channel convolution.
    cache.conv_pre_activation =
        multiChannelConvolution(input, filters);

    // 2. Add convolution biases.
    for (size_t n = 0;
         n < cache.conv_pre_activation.shape[0];
         ++n) {

        for (size_t c = 0;
             c < cache.conv_pre_activation.shape[1];
             ++c) {

            for (size_t h = 0;
                 h < cache.conv_pre_activation.shape[2];
                 ++h) {

                for (size_t w = 0;
                     w < cache.conv_pre_activation.shape[3];
                     ++w) {

                    cache.conv_pre_activation(n, c, h, w)
                        += convBiases[c];
                }
            }
        }
    }

    // 3. ReLU activation.
    cache.relu_output =
        relu(cache.conv_pre_activation);

    // 4. Max pooling.
    PoolingParams pooling;

    pooling.window_h = 2;
    pooling.window_w = 2;
    pooling.stride_h = 2;
    pooling.stride_w = 2;

    MaxPoolingResult pooled =
        maxPool2D(cache.relu_output, pooling);

    cache.pool_output = pooled.output;
    cache.pool_indices = pooled.max_indices;

    // 5. Fully connected layer.
    std::vector<double> logits(
        CLASS_COUNT,
        0.0
    );

    for (size_t c = 0; c < CLASS_COUNT; ++c) {

        logits[c] = denseBiases[c];

        for (size_t j = 0;
             j < DENSE_INPUT;
             ++j) {

            logits[c] +=
                denseWeights[c][j] *
                cache.pool_output.data[j];
        }
    }

    // 6. Softmax.
    cache.probabilities = softmax(logits);

    return cache;
}

void flipHorizontally(
    std::vector<double>& features
) {
    const size_t plane =
        IMAGE_HEIGHT * IMAGE_WIDTH;

    for (size_t c = 0; c < INPUT_CHANNELS; ++c) {

        for (size_t y = 0; y < IMAGE_HEIGHT; ++y) {

            for (size_t x = 0;
                 x < IMAGE_WIDTH / 2;
                 ++x) {

                size_t left =
                    c * plane + y * IMAGE_WIDTH + x;

                size_t right =
                    c * plane +
                    y * IMAGE_WIDTH +
                    (IMAGE_WIDTH - 1 - x);

                std::swap(
                    features[left],
                    features[right]
                );
            }
        }
    }
}

void adjustBrightness(
    std::vector<double>& features,
    double factor
) {
    for (double& value : features) {
        value = std::max(
            0.0,
            std::min(1.0, value * factor)
        );
    }
}

} // anonymous namespace


DogCatModel::DogCatModel()
    : feature_count(
        INPUT_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH
    ),
      initialized(false)
{
}


bool DogCatModel::extractFeatures(
    const std::string& filename,
    std::vector<double>& features
) const
{
    Image image;

    if (!loadImage(filename, image)) {
        std::cerr
            << "Could not load image: "
            << filename << '\n';

        return false;
    }

    int width = image.getWidth();
    int height = image.getHeight();

    if (width <= 0 || height <= 0 ||
        image.getChannels() != 3) {

        std::cerr
            << "Invalid RGB image: "
            << filename << '\n';

        return false;
    }

    const std::vector<Pixel>& pixels =
        image.getPixels();

    if (pixels.size() !=
        static_cast<size_t>(width) *
        static_cast<size_t>(height)) {

        return false;
    }

    std::vector<unsigned char> rgb;

    rgb.reserve(pixels.size() * 3);

    for (const Pixel& pixel : pixels) {
        rgb.push_back(pixel.r);
        rgb.push_back(pixel.g);
        rgb.push_back(pixel.b);
    }

    Tensor input = fromInterleaved(
        rgb.data(),
        rgb.size(),
        static_cast<size_t>(height),
        static_cast<size_t>(width),
        3
    );

    PreprocessConfig config;

    // Preserve all three RGB channels.
    config.target_channels = 3;

    config.target_height = IMAGE_HEIGHT;
    config.target_width = IMAGE_WIDTH;

    config.interpolation =
        Interpolation::Bilinear;

    config.normalization =
        Normalization::Scale;

    Tensor processed = preprocess(
        input,
        config
    );

    if (processed.data.size() != feature_count) {
        return false;
    }

    // Data is stored in NCHW order.
    features = processed.data;

    return true;
}


void DogCatModel::train(
    const std::vector<DogCatSample>& samples,
    size_t epochs,
    double learning_rate
) {
    if (samples.empty()) {
        throw std::invalid_argument(
            "Training dataset is empty."
        );
    }

    if (epochs == 0 ||
        !std::isfinite(learning_rate) ||
        learning_rate <= 0.0) {

        throw std::invalid_argument(
            "Invalid training configuration."
        );
    }

    // Validate the training data.
    size_t classCounts[2] = {0, 0};

    for (const DogCatSample& sample : samples) {

        if (sample.features.size() != feature_count) {
            throw std::invalid_argument(
                "Incorrect training feature size."
            );
        }

        if (sample.label >= CLASS_COUNT) {
            throw std::invalid_argument(
                "Labels must be 0 or 1."
            );
        }

        ++classCounts[sample.label];

        for (double value : sample.features) {

            if (!std::isfinite(value) ||
                value < 0.0 ||
                value > 1.0) {

                throw std::invalid_argument(
                    "Features must be finite values between 0 and 1."
                );
            }
        }
    }

    if (classCounts[0] == 0 ||
        classCounts[1] == 0) {

        throw std::invalid_argument(
            "Training data must contain both cats and dogs."
        );
    }

    // Initialize parameters.
    std::mt19937 generator(42);

    double convLimit = std::sqrt(
        2.0 /
        (INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE)
    );

    std::normal_distribution<double> convDistribution(
        0.0,
        convLimit
    );

    conv_filters = Tensor(
        FILTER_COUNT,
        INPUT_CHANNELS,
        KERNEL_SIZE,
        KERNEL_SIZE
    );

    conv_biases.assign(
        FILTER_COUNT,
        0.0
    );

    for (double& weight : conv_filters.data) {
        weight = convDistribution(generator);
    }

    // Xavier initialization for dense weights.
    double denseLimit = std::sqrt(
        6.0 /
        static_cast<double>(DENSE_INPUT + CLASS_COUNT)
    );

    std::uniform_real_distribution<double> denseDistribution(
        -denseLimit,
        denseLimit
    );

    dense_weights.assign(
        CLASS_COUNT,
        std::vector<double>(DENSE_INPUT)
    );

    dense_biases.assign(
        CLASS_COUNT,
        0.0
    );

    for (size_t c = 0; c < CLASS_COUNT; ++c) {
        for (size_t j = 0; j < DENSE_INPUT; ++j) {
            dense_weights[c][j] =
                denseDistribution(generator);
        }
    }

    initialized = true;

    // Shuffle the samples every epoch.
    std::vector<size_t> indices(samples.size());

    std::iota(
        indices.begin(),
        indices.end(),
        0
    );

    std::mt19937 shuffleGenerator(12345);

    std::uniform_real_distribution<double> augmentationChance(
        0.0,
        1.0
    );

    std::uniform_real_distribution<double> brightnessDistribution(
        0.90,
        1.10
    );

    std::cout
        << "\nTraining CNN classifier...\n";

    std::cout
        << "Architecture: Conv -> ReLU -> MaxPool -> Dense -> Softmax\n";

    std::cout
        << "Training samples: "
        << samples.size() << '\n';

    for (size_t epoch = 0; epoch < epochs; ++epoch) {

        std::shuffle(
            indices.begin(),
            indices.end(),
            shuffleGenerator
        );

        double totalLoss = 0.0;

        for (size_t index : indices) {

            const DogCatSample& sample =
                samples[index];

            // Copy features so augmentation does not
            // modify the original training sample.
            std::vector<double> augmented =
                sample.features;

            // Random horizontal flip.
            if (augmentationChance(generator) < 0.5) {
                flipHorizontally(augmented);
            }

            // Small brightness adjustment.
            if (augmentationChance(generator) < 0.5) {
                adjustBrightness(
                    augmented,
                    brightnessDistribution(generator)
                );
            }

            Tensor input = createInputTensor(
                augmented
            );

            // Forward propagation.
            ForwardCache cache = forwardPass(
                input,
                conv_filters,
                conv_biases,
                dense_weights,
                dense_biases
            );

            // Cross-entropy loss.
            double probability = std::max(
                cache.probabilities[sample.label],
                1e-12
            );

            totalLoss -= std::log(probability);

            // Gradient of softmax + cross entropy.
            std::vector<double> gradLogits =
                cache.probabilities;

            gradLogits[sample.label] -= 1.0;

            // Fully connected gradients.
            std::vector<std::vector<double>> gradDenseWeights(
                CLASS_COUNT,
                std::vector<double>(DENSE_INPUT, 0.0)
            );

            std::vector<double> gradDenseBiases(
                CLASS_COUNT,
                0.0
            );

            Tensor gradPool(
                1,
                FILTER_COUNT,
                POOL_HEIGHT,
                POOL_WIDTH
            );

            gradPool.fill(0.0);

            for (size_t c = 0; c < CLASS_COUNT; ++c) {

                gradDenseBiases[c] =
                    gradLogits[c];

                for (size_t j = 0; j < DENSE_INPUT; ++j) {

                    gradDenseWeights[c][j] =
                        gradLogits[c] *
                        cache.pool_output.data[j];

                    gradPool.data[j] +=
                        dense_weights[c][j] *
                        gradLogits[c];
                }
            }

            // Backpropagation through max pooling.
            Tensor gradRelu(
                1,
                FILTER_COUNT,
                CONV_HEIGHT,
                CONV_WIDTH
            );

            gradRelu.fill(0.0);

            for (size_t i = 0;
                 i < cache.pool_indices.size();
                 ++i) {

                size_t originalIndex =
                    cache.pool_indices[i];

                gradRelu.data[originalIndex] +=
                    gradPool.data[i];
            }

            // Backpropagation through ReLU.
            Tensor gradConv = gradRelu;

            for (size_t i = 0;
                 i < gradConv.data.size();
                 ++i) {

                if (cache.conv_pre_activation.data[i] <= 0.0) {
                    gradConv.data[i] = 0.0;
                }
            }

            // Convolution kernel gradients.
            Tensor gradFilters =
                convolutionKernelGradient(
                    cache.input,
                    conv_filters,
                    gradConv
                );

            // Convolution bias gradients.
            std::vector<double> gradConvBiases(
                FILTER_COUNT,
                0.0
            );

            for (size_t c = 0; c < FILTER_COUNT; ++c) {

                for (size_t h = 0; h < CONV_HEIGHT; ++h) {
                    for (size_t w = 0; w < CONV_WIDTH; ++w) {

                        gradConvBiases[c] +=
                            gradConv(0, c, h, w);
                    }
                }
            }

            // Update convolution filters.
            for (size_t i = 0;
                 i < conv_filters.data.size();
                 ++i) {

                conv_filters.data[i] -= learning_rate *
                    (
                        gradFilters.data[i] +
                        WEIGHT_DECAY * conv_filters.data[i]
                    );
            }

            // Update convolution biases.
            for (size_t c = 0; c < FILTER_COUNT; ++c) {
                conv_biases[c] -=
                    learning_rate * gradConvBiases[c];
            }

            // Update dense weights and biases.
            for (size_t c = 0; c < CLASS_COUNT; ++c) {

                dense_biases[c] -=
                    learning_rate * gradDenseBiases[c];

                for (size_t j = 0; j < DENSE_INPUT; ++j) {

                    dense_weights[c][j] -= learning_rate *
                        (
                            gradDenseWeights[c][j] +
                            WEIGHT_DECAY * dense_weights[c][j]
                        );
                }
            }
        }

        double averageLoss =
            totalLoss / static_cast<double>(samples.size());

        if ((epoch + 1) % 5 == 0 || epoch == 0) {

            std::cout
                << "Epoch "
                << epoch + 1
                << " / "
                << epochs
                << " | Loss: "
                << averageLoss
                << '\n';
        }
    }

    std::cout
        << "\nCNN training completed.\n";
}


std::vector<double> DogCatModel::predictProbabilities(
    const std::vector<double>& features
) const {
    if (!initialized) {
        throw std::runtime_error(
            "Model has not been trained or loaded."
        );
    }

    Tensor input = createInputTensor(features);

    ForwardCache cache = forwardPass(
        input,
        conv_filters,
        conv_biases,
        dense_weights,
        dense_biases
    );

    return cache.probabilities;
}


size_t DogCatModel::predict(
    const std::vector<double>& features
) const {
    std::vector<double> probabilities =
        predictProbabilities(features);

    return probabilities[1] > probabilities[0] ? 1 : 0;
}


bool DogCatModel::save(
    const std::string& filename
) const {
    if (!initialized) {
        return false;
    }

    std::ofstream file(filename);

    if (!file.is_open()) {
        return false;
    }

    file.precision(
        std::numeric_limits<double>::max_digits10
    );

    file << "DOGCAT_CNN_V2\n";

    file << feature_count << '\n';

    for (double value : conv_filters.data) {
        file << value << ' ';
    }

    file << '\n';

    for (double value : conv_biases) {
        file << value << ' ';
    }

    file << '\n';

    for (const auto& row : dense_weights) {
        for (double value : row) {
            file << value << ' ';
        }

        file << '\n';
    }

    for (double value : dense_biases) {
        file << value << ' ';
    }

    file << '\n';

    return file.good();
}


bool DogCatModel::load(
    const std::string& filename
) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        return false;
    }

    std::string version;

    if (!(file >> version) ||
        version != "DOGCAT_CNN_V2") {

        return false;
    }

    size_t loadedFeatureCount = 0;

    if (!(file >> loadedFeatureCount) ||
        loadedFeatureCount != feature_count) {

        return false;
    }

    Tensor loadedFilters(
        FILTER_COUNT,
        INPUT_CHANNELS,
        KERNEL_SIZE,
        KERNEL_SIZE
    );

    std::vector<double> loadedConvBiases(
        FILTER_COUNT
    );

    std::vector<std::vector<double>> loadedDenseWeights(
        CLASS_COUNT,
        std::vector<double>(DENSE_INPUT)
    );

    std::vector<double> loadedDenseBiases(
        CLASS_COUNT
    );

    for (double& value : loadedFilters.data) {
        if (!(file >> value) || !std::isfinite(value)) {
            return false;
        }
    }

    for (double& value : loadedConvBiases) {
        if (!(file >> value) || !std::isfinite(value)) {
            return false;
        }
    }

    for (auto& row : loadedDenseWeights) {
        for (double& value : row) {
            if (!(file >> value) || !std::isfinite(value)) {
                return false;
            }
        }
    }

    for (double& value : loadedDenseBiases) {
        if (!(file >> value) || !std::isfinite(value)) {
            return false;
        }
    }

    conv_filters = loadedFilters;
    conv_biases = loadedConvBiases;
    dense_weights = loadedDenseWeights;
    dense_biases = loadedDenseBiases;

    initialized = true;

    return true;
}


size_t DogCatModel::getFeatureCount() const {
    return feature_count;
}

}