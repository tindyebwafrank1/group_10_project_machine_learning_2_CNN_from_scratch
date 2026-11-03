#ifndef DOG_CAT_MODEL_HPP
#define DOG_CAT_MODEL_HPP

#include <vector>
#include <string>
#include <cstddef>

#include "image_loader.hpp"

using namespace std;

namespace cnn {

struct DogCatSample {
    vector<double> features;
    size_t label; // 0 = CAT, 1 = DOG
};

class DogCatModel {
private:
    vector<vector<double>> weights;
    vector<double> biases;
    size_t feature_count;

public:
    DogCatModel();

    bool extractFeatures(
        const string& filename,
        vector<double>& features
    ) const;

    void train(
        const vector<DogCatSample>& samples,
        size_t epochs,
        double learning_rate
    );

    size_t predict(const vector<double>& features) const;

    vector<double> predictProbabilities(
        const vector<double>& features
    ) const;

    bool save(const string& filename) const;

    bool load(const string& filename);

    size_t getFeatureCount() const;
};

}

#endif