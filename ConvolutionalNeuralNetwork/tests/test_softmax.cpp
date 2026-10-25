#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <limits>

#include "fully_connected.hpp"
#include "softmax.hpp"

using namespace std;

static bool close(double a, double b, double tol = 1e-9)
{
    return fabs(a - b) < tol;
}

int main()
{
    cnn::Softmax sm;

    // Test 1: Known Softmax probabilities
    // Scores [2, 1, 0.1]
    // Expected probabilities approximately [0.659001, 0.242433, 0.098566]
    Tensor a(1, 3, 1, 1);
    a.data = {2.0, 1.0, 0.1};

    Tensor p = sm.forward(a);

    assert(close(p.data[0], 0.659001, 1e-6));
    assert(close(p.data[1], 0.242433, 1e-6));
    assert(close(p.data[2], 0.098566, 1e-6));
    assert(p.shape == a.shape);

    // The original input must not be modified
    assert(a.data[0] == 2.0);

    // Test 2: Batch of two samples
    Tensor b(2, 4, 1, 1);

    for (size_t i = 0; i < b.data.size(); ++i)
    {
        b.data[i] = 0.7 * i - 2.0;
    }

    Tensor q = sm.forward(b);

    for (size_t n = 0; n < 2; ++n)
    {
        double sum = 0.0;

        for (size_t c = 0; c < 4; ++c)
        {
            double probability = q.data[n * 4 + c];

            assert(probability > 0.0);
            assert(probability < 1.0);

            sum += probability;
        }

        assert(close(sum, 1.0, 1e-12));
    }

    // Test 3: Large positive scores
    // Stable Softmax should avoid overflow
    Tensor h(1, 3, 1, 1);
    h.data = {1000.0, 1001.0, 1002.0};

    Tensor r = sm.forward(h);

    for (double value : r.data)
    {
        assert(isfinite(value));
    }

    assert(close(r.data[2], 0.665241, 1e-6));

    // Test 4: Large negative scores
    Tensor m(1, 3, 1, 1);
    m.data = {-1000.0, -1001.0, -1002.0};

    Tensor t = sm.forward(m);

    for (double value : t.data)
    {
        assert(isfinite(value));
    }

    assert(close(t.data[0], 0.665241, 1e-6));

    // Test 5: Adding the same constant to every score
    // must not change the probabilities
    Tensor x(1, 3, 1, 1);
    Tensor y(1, 3, 1, 1);

    x.data = {0.0, 1.0, 2.0};
    y.data = {50.0, 51.0, 52.0};

    Tensor px = sm.forward(x);
    Tensor py = sm.forward(y);

    for (size_t i = 0; i < 3; ++i)
    {
        assert(close(px.data[i], py.data[i], 1e-12));
    }

    // Test 6: One class must have probability 1
    Tensor one(1, 1, 1, 1);
    one.data = {7.0};

    Tensor one_result = sm.forward(one);

    assert(close(one_result.data[0], 1.0, 1e-12));

    // Test 7: output() stores the most recent result
    sm.forward(a);

    assert(sm.output().size() == 3);
    assert(close(sm.output()[0], 0.659001, 1e-6));

    // Test 8: Reject incorrect tensor shape
    bool thrown = false;

    try
    {
        Tensor bad(1, 2, 2, 2);
        sm.forward(bad);
    }
    catch (const invalid_argument&)
    {
        thrown = true;
    }

    assert(thrown);

    // Test 9: Reject NaN input
    thrown = false;

    try
    {
        Tensor bad(1, 2, 1, 1);
        bad.data[0] = numeric_limits<double>::quiet_NaN();

        sm.forward(bad);
    }
    catch (const invalid_argument&)
    {
        thrown = true;
    }

    assert(thrown);

    // Test 10: Reject infinite input
    thrown = false;

    try
    {
        Tensor bad(1, 2, 1, 1);
        bad.data[1] = numeric_limits<double>::infinity();

        sm.forward(bad);
    }
    catch (const invalid_argument&)
    {
        thrown = true;
    }

    assert(thrown);

    // Test 11: Integration with FullyConnectedLayer
// Input = [1, 2]
// Expected scores = [1, 2, 3]

// Create a fully connected layer with 2 inputs and 3 outputs.
cnn::FullyConnectedLayer fc(
    {
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    },
    {0.0, 0.0, 0.0}
);

// Prepare the input vector.
vector<double> input_values = {1.0, 2.0};

// Calculate the fully connected layer's output.
vector<double> scores_values = fc.forward(input_values);

// Store the scores in the project's 4D Tensor.
Tensor scores(1, 3, 1, 1);
scores.data = scores_values;

// Apply Softmax.
Tensor probabilities = sm.forward(scores);

// Check that the probabilities sum to 1.
double sum = probabilities.data[0]
           + probabilities.data[1]
           + probabilities.data[2];

assert(close(sum, 1.0, 1e-12));

// Scores are ordered 1 < 2 < 3,
// so their Softmax probabilities must have the same order.
assert(probabilities.data[2] > probabilities.data[1]);
assert(probabilities.data[1] > probabilities.data[0]);

cout << "All softmax tests passed" << endl;

return 0;
}
