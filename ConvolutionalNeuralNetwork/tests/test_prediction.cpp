
#include <iostream>
#include <vector>
#include <stdexcept>
#include <limits>

#include "prediction.hpp"

using namespace std;

int main()
{
    try {
        // Test 1: Identify the class with the highest score.
        vector<double> scores1 = {0.15, 0.70, 0.15};

        if (cnn::argmax(scores1) != 1) {
            throw runtime_error("Test 1 failed");
        }

        cout << "Test 1 passed: highest score selected\n";


        // Test 2: The highest score is at the first index.
        vector<double> scores2 = {0.90, 0.05, 0.05};

        if (cnn::argmax(scores2) != 0) {
            throw runtime_error("Test 2 failed");
        }

        cout << "Test 2 passed: first class selected\n";


        // Test 3: The highest score is at the last index.
        vector<double> scores3 = {0.10, 0.20, 0.80};

        if (cnn::predict(scores3) != 2) {
            throw runtime_error("Test 3 failed");
        }

        cout << "Test 3 passed: last class selected\n";


        // Test 4: Equal scores select the first maximum.
        vector<double> scores4 = {0.60, 0.60, 0.20};

        if (cnn::argmax(scores4) != 0) {
            throw runtime_error("Test 4 failed");
        }

        cout << "Test 4 passed: ties handled correctly\n";


        // Test 5: A single class must return index zero.
        vector<double> scores5 = {0.75};

        if (cnn::predict(scores5) != 0) {
            throw runtime_error("Test 5 failed");
        }

        cout << "Test 5 passed: single class handled\n";


        // Test 6: An empty vector must be rejected.
        bool exception_thrown = false;

        try {
            vector<double> empty_scores;
            cnn::argmax(empty_scores);
        }
        catch (const invalid_argument&) {
            exception_thrown = true;
        }

        if (!exception_thrown) {
            throw runtime_error("Test 6 failed");
        }

        cout << "Test 6 passed: empty input rejected\n";


        // Test 7: Non-finite scores must be rejected.
        exception_thrown = false;

        try {
            vector<double> invalid_scores = {
                0.2, numeric_limits<double>::quiet_NaN()
            };

            cnn::argmax(invalid_scores);
        }
        catch (const invalid_argument&) {
            exception_thrown = true;
        }

        if (!exception_thrown) {
            throw runtime_error("Test 7 failed");
        }

        cout << "Test 7 passed: invalid score rejected\n";

        cout << "\nAll 7 prediction tests passed.\n";
    }
    catch (const exception& e) {
        cerr << e.what() << '\n';
        return 1;
    }

    return 0;
}
