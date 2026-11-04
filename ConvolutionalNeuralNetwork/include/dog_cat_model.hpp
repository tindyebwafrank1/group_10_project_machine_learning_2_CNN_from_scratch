#ifndef DOG_CAT_MODEL_HPP
#define DOG_CAT_MODEL_HPP

#include <vector>
#include <string>
#include <cstddef>

#include "image_loader.hpp"
#include "tensor.hpp"

namespace cnn {

struct DogCatSample {
    std::vector<double> features;
    size_t label; // 0 = CAT, 1 = DOG
};

class DogCatModel {
private:
    // Convolution layer: 8 filters, 3 input channels, 3x3 kernels.
    Tensor conv_filters;
    std::vector<double> conv_biases;

    // Fully connected output layer.
    std::vector<std::vector<double>> dense_weights;
    std::vector<double> dense_biases;

    size_t feature_count;
    bool initialized;

public:
    DogCatModel();

    bool extractFeatures(
        const std::string& filename,
        std::vector<double>& features
    ) const;

    void train(
        const std::vector<DogCatSample>& samples,
        size_t epochs,
        double learning_rate
    );

    size_t predict(
        const std::vector<double>& features
    ) const;

    std::vector<double> predictProbabilities(
        const std::vector<double>& features
    ) const;

    bool save(const std::string& filename) const;

    bool load(const std::string& filename);

    size_t getFeatureCount() const;
};

}

#endif