
#include "../include/dog_cat_model.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cctype>
#include <windows.h>

using namespace std;

// Load JPG and JPEG images from a directory.
void loadImages(
    const string& folder,
    size_t label,
    cnn::DogCatModel& model,
    vector<cnn::DogCatSample>& samples
) {
    DWORD attributes = GetFileAttributesA(folder.c_str());

    if (attributes == INVALID_FILE_ATTRIBUTES ||
        !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        throw runtime_error("Folder not found: " + folder);
    }

    string searchPath = folder + "\\*";

    WIN32_FIND_DATAA fileData;

    HANDLE handle = FindFirstFileA(
        searchPath.c_str(),
        &fileData
    );

    if (handle == INVALID_HANDLE_VALUE) {
        throw runtime_error("Could not read folder: " + folder);
    }

    do {
        string filename = fileData.cFileName;

        // Skip directories.
        if (fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            continue;
        }

        // Get file extension.
        size_t dotPosition = filename.find_last_of('.');

        if (dotPosition == string::npos) {
            continue;
        }

        string extension = filename.substr(dotPosition);

        for (char& ch : extension) {
            ch = static_cast<char>(
                tolower(static_cast<unsigned char>(ch))
            );
        }

        // Accept JPG and JPEG files.
        if (extension != ".jpg" && extension != ".jpeg") {
            continue;
        }

        string imagePath = folder + "\\" + filename;

        vector<double> features;

        if (model.extractFeatures(imagePath, features)) {

            cnn::DogCatSample sample;

            sample.features = features;
            sample.label = label;

            samples.push_back(sample);

            cout << "Loaded " << imagePath << '\n';
        }
        else {
            cout << "Warning: Could not process "
                 << imagePath << '\n';
        }

    } while (FindNextFileA(handle, &fileData));

    FindClose(handle);
}

int main() {

    try {

        cout << "====================================\n";
        cout << " GROUP 10 DOG AND CAT CLASSIFIER\n";
        cout << "====================================\n";

        cnn::DogCatModel model;

        vector<cnn::DogCatSample> trainingSamples;
        vector<cnn::DogCatSample> testingSamples;

        // Load training images.
        cout << "\nLoading training images...\n";

        loadImages(
            "data/dogs_cats/train/cat",
            0,
            model,
            trainingSamples
        );

        loadImages(
            "data/dogs_cats/train/dog",
            1,
            model,
            trainingSamples
        );

        // Load testing images.
        cout << "\nLoading testing images...\n";

        loadImages(
            "data/dogs_cats/test/cat",
            0,
            model,
            testingSamples
        );

        loadImages(
            "data/dogs_cats/test/dog",
            1,
            model,
            testingSamples
        );

        // Validate datasets.
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

        cout << "\nTraining images: "
             << trainingSamples.size() << '\n';

        cout << "Testing images: "
             << testingSamples.size() << '\n';

        // Train the model.
        cout << "\nStarting training...\n";

        model.train(
            trainingSamples,
            100,
            0.01
        );

        // Save trained model.
        if (!model.save("dog_cat_model.txt")) {
            throw runtime_error(
                "Could not save trained model."
            );
        }

        cout << "\nModel saved to dog_cat_model.txt\n";

        // Evaluate model.
        cout << "\nTesting model...\n";

        size_t correct = 0;

        for (size_t i = 0; i < testingSamples.size(); ++i) {

            size_t predicted =
                model.predict(testingSamples[i].features);

            size_t actual =
                testingSamples[i].label;

            string predictedName =
                predicted == 0 ? "CAT" : "DOG";

            string actualName =
                actual == 0 ? "CAT" : "DOG";

            cout << "Image " << i + 1
                 << " | Actual: " << actualName
                 << " | Predicted: " << predictedName;

            if (predicted == actual) {

                ++correct;

                cout << " | CORRECT\n";
            }
            else {

                cout << " | INCORRECT\n";
            }
        }

        // Calculate accuracy.
        double accuracy =
            100.0 * correct / testingSamples.size();

        cout << "\n====================================\n";
        cout << "FINAL TEST RESULTS\n";
        cout << "====================================\n";

        cout << "Correct predictions: "
             << correct << '\n';

        cout << "Total test images: "
             << testingSamples.size() << '\n';

        cout << "Accuracy: "
             << accuracy << "%\n";

        cout << "\n====================================\n";
        cout << "CLASSIFICATION COMPLETED\n";
        cout << "====================================\n";

    }
    catch (const exception& e) {

        cerr << "\nERROR: " << e.what() << '\n';

        return 1;
    }

    return 0;
}
