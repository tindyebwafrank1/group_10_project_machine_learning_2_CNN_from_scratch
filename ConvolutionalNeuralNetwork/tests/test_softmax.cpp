
#include "softmax.hpp"
#include "fully_connected.hpp"
#include "tensor.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>

using namespace std;

bool close(double a, double b, double tolerance = 1e-9)
{
    return abs(a - b) < tolerance;
}

int main()
{
    cnn::Softmax sm;

    // Test 1: Basic softmax calculation.
    {
        Tensor input(1, 3, 1, 1);
        input.data = {1.0, 2.0, 3.0};

        Tensor output = sm.forward(input);

        double sum = 0.0;
        for (double value : output.data)
        {
            sum += value;
        }

        assert(close(sum, 1.0));
    }

    // Test 2: All equal values produce equal probabilities.
    {
        Tensor input(1, 3, 1, 1);
        input.data = {2.0, 2.0, 2.0};

        Tensor output = sm.forward(input);

        assert(close(output.data[0], 1.0 / 3.0));
        assert(close(output.data[1], 1.0 / 3.0));
        assert(close(output.data[2], 1.0 / 3.0));
    }

    // Test 3: Larger input gets a larger probability.
    {
        Tensor input(1, 3, 1, 1);
        input.data = {1.0, 2.0, 3.0};

        Tensor output = sm.forward(input);

        assert(output.data[2] > output.data[1]);
        assert(output.data[1] > output.data[0]);
    }

    // Test 4: Single input value produces probability 1.
    {
        Tensor input(1, 1, 1, 1);
        input.data = {5.0};

        Tensor output = sm.forward(input);

        assert(close(output.data[0], 1.0));
    }

    // Test 5: Large values remain numerically stable.
    {
        Tensor input(1, 3, 1, 1);
        input.data = {1000.0, 1001.0, 1002.0};

        Tensor output = sm.forward(input);

        double sum = 0.0;
        for (double value : output.data)
        {
            sum += value;
        }

        assert(close(sum, 1.0));
    }

    // Test 6: Negative values.
    {
        Tensor input(1, 3, 1, 1);
        input.data = {-1.0, -2.0, -3.0};

        Tensor output = sm.forward(input);

        double sum = 0.0;
        for (double value : output.data)
        {
            sum += value;
        }

        assert(close(sum, 1.0));
    }

    // Test 7: Output probabilities are between 0 and 1.
    {
        Tensor input(1, 4, 1, 1);
        input.data = {0.5, 1.5, 2.5, 3.5};

        Tensor output = sm.forward(input);

        for (double value : output.data)
        {
            assert(value >= 0.0);
            assert(value <= 1.0);
        }
    }

    // Test 8: Output has the same shape as input.
    {
        Tensor input(1, 3, 1, 1);
        input.data = {1.0, 2.0, 3.0};

        Tensor output = sm.forward(input);

        assert(output.shape == input.shape);
    }

    // Test 9: Empty or invalid input should be rejected.
    {
        bool exception_thrown = false;

        try
        {
            Tensor input(1, 0, 1, 1);
            sm.forward(input);
        }
        catch (const exception&)
        {
            exception_thrown = true;
        }

        assert(exception_thrown);
    }

    // Test 10: Fully connected output can be passed to Softmax.
    {
        cnn::FullyConnected fc(2, 3);

        fc.setWeights({
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0
        });

        fc.setBias({0.0, 0.0, 0.0});

        // Input values: [1, 2]
        Tensor input(1, 2, 1, 1);
        input.data = {1.0, 2.0};

        // Expected fully connected scores: [1, 2, 3]
        Tensor scores = fc.forward(input);

        Tensor probabilities = sm.forward(scores);

        double sum = 0.0;
        for (double value : probabilities.data)
        {
            sum += value;
        }

        assert(close(sum, 1.0));
        assert(probabilities.data[2] > probabilities.data[1]);
        assert(probabilities.data[1] > probabilities.data[0]);
    }

    cout << "All softmax tests passed" << endl;

    return 0;
}
