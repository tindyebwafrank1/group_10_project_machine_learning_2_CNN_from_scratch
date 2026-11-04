#include "../include/dog_cat_model.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <windows.h>

using namespace std;

void loadImages(
    const string& folder,
    size_t label,
    cnn::DogCatModel& model,
    vector<cnn::DogCatSample>& samples
) {
    DWORD attributes = GetFileAttributesA(folder.c_str());

    if (attributes == INVALID_FILE_ATTRIBUTES ||
        !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {

        throw runtime_error(
            "Folder not found: " + folder
        );
    }

    string searchPath = folder + "\\*";

    WIN32_FIND_DATAA fileData;

    HANDLE handle = FindFirstFileA(
        searchPath.c_str(),
        &fileData
    );

    if (handle == INVALID_HANDLE_VALUE) {
        throw runtime_error(
            "Could not read folder: " + folder
        );
    }

    do {
        string filename = fileData.cFileName;

        if (fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            continue;
        }

        size_t dotPosition =
            filename.find_last_of('.');

        if (dotPosition == string::npos) {
            continue;
        }

        string extension =
            filename.substr(dotPosition);

        transform(
            extension.begin(),
            extension.end(),
            extension.begin(),
            [](unsigned char ch) {
                return static_cast<char>(tolower(ch));
            }
        );

        // Supported formats.
        if (extension != ".jpg" &&
            extension != ".jpeg" &&
            extension != ".png" &&
            extension != ".bmp") {

            continue;
        }

        string imagePath =
            folder + "\\" + filename;

        vector<double> features;

        if (model.extractFeatures(imagePath, features)) {

            cnn::DogCatSample sample;

            sample.features = features;
            sample.label = label;

            samples.push_back(sample);

            cout << "Loaded: "
                 << imagePath << '\n';
        }
        else {
            cout << "Warning: Could not process "
                 << imagePath << '\n';
        }

    } while (FindNextFileA(handle, &fileData));

    FindClose(handle);
}


int main(int argc, char* argv[]) {

    try {

        cout << "\n";
        cout << "       GROUP 10 DOG AND CAT CNN\n";
        cout << "\n";

        // Default dataset paths.
        string trainCatPath =
            "data/dogs_cats/train/cat";

        string trainDogPath =
            "data/dogs_cats/train/dog";

        string testCatPath =
            "data/dogs_cats/test/cat";

        string testDogPath =
            "data/dogs_cats/test/dog";

        // Optional command-line arguments.
        if (argc > 1) {
            trainCatPath = string(argv[1]) + "\\cat";
            trainDogPath = string(argv[1]) + "\\dog";
        }

        if (argc > 2) {
            testCatPath = string(argv[2]) + "\\cat";
            testDogPath = string(argv[2]) + "\\dog";
        }

        size_t epochs = 50;

        if (argc > 3) {
            epochs = static_cast<size_t>(
                atoi(argv[3])
            );
        }

        if (epochs == 0) {
            throw runtime_error(
                "Epoch count must be greater than zero."
            );
        }

        const double learningRate = 0.001;

        cnn::DogCatModel model;

        vector<cnn::DogCatSample> trainingSamples;
        vector<cnn::DogCatSample> testingSamples;

        // LOAD TRAINING DATA

        cout << "\nLoading training images...\n";

        loadImages(
            trainCatPath,
            0,
            model,
            trainingSamples
        );

        loadImages(
            trainDogPath,
            1,
            model,
            trainingSamples
        );

        // LOAD TESTING DATA
        

        cout << "\nLoading testing images...\n";

        loadImages(
            testCatPath,
            0,
            model,
            testingSamples
        );

        loadImages(
            testDogPath,
            1,
            model,
            testingSamples
        );

        if (trainingSamples.empty()) {
            throw runtime_error(
                "No training images were loaded."
            );
        }

        if (testingSamples.empty()) {
            throw runtime_error(
                "No testing images were loaded."
            );
        }

        cout << "\n\n";
        cout << "DATASET SUMMARY\n";
        cout << "\n";

        size_t trainCats = 0;
        size_t trainDogs = 0;

        size_t testCats = 0;
        size_t testDogs = 0;

        for (const auto& sample : trainingSamples) {
            if (sample.label == 0) {
                ++trainCats;
            }
            else {
                ++trainDogs;
            }
        }

        for (const auto& sample : testingSamples) {
            if (sample.label == 0) {
                ++testCats;
            }
            else {
                ++testDogs;
            }
        }

        cout << "Training cats: " << trainCats << '\n';
        cout << "Training dogs: " << trainDogs << '\n';

        cout << "Testing cats: " << testCats << '\n';
        cout << "Testing dogs: " << testDogs << '\n';

        cout << "Total training images: "
             << trainingSamples.size() << '\n';

        cout << "Total testing images: "
             << testingSamples.size() << '\n';

        
        // TRAIN CNN
        

        cout << "\n\n";
        cout << "CNN TRAINING\n";
        cout << "\n";

        cout << "Epochs: " << epochs << '\n';
        cout << "Learning rate: " << learningRate << '\n';

        model.train(
            trainingSamples,
            epochs,
            learningRate
        );

    
        // SAVE MODEL
        

        if (!model.save("dog_cat_model.txt")) {
            throw runtime_error(
                "Could not save the trained CNN model."
            );
        }

        cout << "\nModel saved successfully.\n";

        
        // EVALUATION
        

        cout << "\n========================================\n";
        cout << "TESTING CNN CLASSIFIER\n";
        cout << "\n";

        size_t confusion[2][2] = {
            {0, 0},
            {0, 0}
        };

        size_t correct = 0;

        for (size_t i = 0;
             i < testingSamples.size();
             ++i) {

            const auto& sample =
                testingSamples[i];

            vector<double> probabilities =
                model.predictProbabilities(
                    sample.features
                );

            size_t predicted =
                probabilities[1] > probabilities[0]
                ? 1 : 0;

            size_t actual = sample.label;

            ++confusion[actual][predicted];

            string predictedName =
                predicted == 0 ? "CAT" : "DOG";

            string actualName =
                actual == 0 ? "CAT" : "DOG";

            cout << "\nImage " << i + 1;

            cout << " | Actual: "
                 << actualName;

            cout << " | Predicted: "
                 << predictedName;

            cout << " | Cat probability: "
                 << probabilities[0] * 100.0 << "%";

            cout << " | Dog probability: "
                 << probabilities[1] * 100.0 << "%";

            if (predicted == actual) {
                ++correct;
                cout << " | CORRECT";
            }
            else {
                cout << " | INCORRECT";
            }

            cout << '\n';
        }

        
        // CALCULATE METRICS
        

        double accuracy =
            100.0 * correct /
            static_cast<double>(testingSamples.size());

        cout << "\n========================================\n";
        cout << "CONFUSION MATRIX\n";
        cout << "\n";

        cout << "                 Predicted CAT  Predicted DOG\n";

        cout << "Actual CAT          "
             << confusion[0][0]
             << "               "
             << confusion[0][1] << '\n';

        cout << "Actual DOG          "
             << confusion[1][0]
             << "               "
             << confusion[1][1] << '\n';

        cout << "\n========================================\n";
        cout << "CLASSIFICATION METRICS\n";
        cout << "\n";

        cout << "Accuracy: "
             << accuracy << "%\n";

        for (size_t c = 0; c < 2; ++c) {

            size_t truePositive =
                confusion[c][c];

            size_t falsePositive =
                confusion[1 - c][c];

            size_t falseNegative =
                confusion[c][1 - c];

            double precision =
                truePositive + falsePositive == 0
                ? 0.0
                : static_cast<double>(truePositive) /
                  (truePositive + falsePositive);

            double recall =
                truePositive + falseNegative == 0
                ? 0.0
                : static_cast<double>(truePositive) /
                  (truePositive + falseNegative);

            double f1 =
                precision + recall == 0.0
                ? 0.0
                : 2.0 * precision * recall /
                  (precision + recall);

            cout << "\n"
                 << (c == 0 ? "CAT" : "DOG")
                 << " METRICS\n";

            cout << "Precision: "
                 << precision * 100.0 << "%\n";

            cout << "Recall: "
                 << recall * 100.0 << "%\n";

            cout << "F1-score: "
                 << f1 * 100.0 << "%\n";
        }

        cout << "\n========================================\n";
        cout << "CLASSIFICATION COMPLETED\n";
        cout << "\n";

    }
    catch (const exception& e) {

        cerr << "\nERROR: "
             << e.what() << '\n';

        return 1;
    }

    return 0;
}