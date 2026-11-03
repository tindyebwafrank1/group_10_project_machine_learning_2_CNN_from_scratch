#include "../include/dog_cat_model.hpp"

#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage:\n";
        cout << "dog_cat_predict.exe path_to_image.jpg\n";
        return 1;
    }

    try {
        cnn::DogCatModel model;

        if (!model.load("dog_cat_model.txt")) {
            cerr << "Could not load dog_cat_model.txt\n";
            cerr << "Train the model first.\n";
            return 1;
        }

        vector<double> features;

        if (!model.extractFeatures(argv[1], features)) {
            cerr << "Could not process the image.\n";
            return 1;
        }

        vector<double> probabilities =
            model.predictProbabilities(features);

        size_t prediction = model.predict(features);

        cout << "\n====================================\n";
        cout << " DOG AND CAT PREDICTION\n";
        cout << "====================================\n";

        cout << "Image: " << argv[1] << '\n';

        cout << "Prediction: "
             << (prediction == 0 ? "CAT" : "DOG") << '\n';

        cout << "Cat probability: "
             << probabilities[0] * 100.0 << "%\n";

        cout << "Dog probability: "
             << probabilities[1] * 100.0 << "%\n";

    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }

    return 0;
}