
#include "flatten.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

int main() {
    // TEST 1: Flatten one sample with two channels.
    Tensor input(1, 2, 2, 2);

    // Channel 0.
    input(0, 0, 0, 0) = 1.0;
    input(0, 0, 0, 1) = 2.0;
    input(0, 0, 1, 0) = 3.0;
    input(0, 0, 1, 1) = 4.0;

    // Channel 1.
    input(0, 1, 0, 0) = 5.0;
    input(0, 1, 0, 1) = 6.0;
    input(0, 1, 1, 0) = 7.0;
    input(0, 1, 1, 1) = 8.0;

    vector<vector<double>> output = cnn::flatten(input);

    assert(output.size() == 1);
    assert(output[0].size() == 8);

    vector<double> expected = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0
    };

    for (size_t i = 0; i < expected.size(); ++i) {
        assert(abs(output[0][i] - expected[i]) < 1e-9);
    }

    // TEST 2: Keep two samples separate.
    Tensor batch(2, 1, 1, 2);

    batch(0, 0, 0, 0) = 10.0;
    batch(0, 0, 0, 1) = 20.0;
    batch(1, 0, 0, 0) = 30.0;
    batch(1, 0, 0, 1) = 40.0;

    vector<vector<double>> batchOutput = cnn::flatten(batch);

    assert(batchOutput.size() == 2);
    assert(batchOutput[0].size() == 2);
    assert(batchOutput[1].size() == 2);

    assert(batchOutput[0][0] == 10.0);
    assert(batchOutput[0][1] == 20.0);
    assert(batchOutput[1][0] == 30.0);
    assert(batchOutput[1][1] == 40.0);

    cout <<endl<< "All flattening tests passed!" << endl<<endl;

    return 0;
}
