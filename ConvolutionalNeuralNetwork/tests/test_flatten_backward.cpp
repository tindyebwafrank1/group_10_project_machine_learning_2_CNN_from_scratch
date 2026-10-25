#include "flatten_backward.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

int main()
{
    // Create an original tensor with shape (1, 1, 2, 3).
    Tensor input(1, 1, 2, 3);

    // Six gradient values returned from the fully connected layer.
    vector<double> flattened_gradient = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };

    // Restore the gradient to the original 4D tensor shape.
    Tensor gradient =
        cnn::unflattenGradient(flattened_gradient, input);

    // Check that the original shape has been restored.
    assert(gradient.shape.size() == 4);
    assert(gradient.shape[0] == 1);
    assert(gradient.shape[1] == 1);
    assert(gradient.shape[2] == 2);
    assert(gradient.shape[3] == 3);

    // Check that every gradient value is preserved.
    for (size_t i = 0; i < flattened_gradient.size(); ++i) {
        assert(gradient.data[i] == flattened_gradient[i]);
    }

    // Check that an incorrectly sized gradient is rejected.
    bool threw = false;

    try {
        cnn::unflattenGradient(
            {1.0, 2.0},
            input
        );
    }
    catch (const invalid_argument&) {
        threw = true;
    }

    assert(threw);

    cout << "All flatten backward tests passed." << endl;

    return 0;
}