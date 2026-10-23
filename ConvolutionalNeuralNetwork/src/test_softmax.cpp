// Unit tests for the Softmax layer
#include <cassert>
#include <cmath>
#include <iostream>
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

    // known values: [2, 1, 0.1] -> [0.659001, 0.242433, 0.098566]
    cnn::Tensor a(1, 3, 1, 1);
    a.data = {2.0, 1.0, 0.1};
    cnn::Tensor p = sm.forward(a);
    assert(close(p.data[0], 0.659001, 1e-6));
    assert(close(p.data[1], 0.242433, 1e-6));
    assert(close(p.data[2], 0.098566, 1e-6));
    assert(p.shape == a.shape);
    assert(a.data[0] == 2.0); // input not modified

    // each sample sums to 1 (batch of 2)
    cnn::Tensor b(2, 4, 1, 1);
    for (size_t i = 0; i < b.data.size(); ++i) b.data[i] = 0.7 * i - 2.0;
    cnn::Tensor q = sm.forward(b);
    for (size_t n = 0; n < 2; ++n)
    {
        double s = 0.0;
        for (size_t c = 0; c < 4; ++c)
        {
            assert(q.data[n * 4 + c] > 0.0 && q.data[n * 4 + c] < 1.0);
            s += q.data[n * 4 + c];
        }
        assert(close(s, 1.0, 1e-12));
    }

    // large positive inputs: no overflow
    cnn::Tensor h(1, 3, 1, 1);
    h.data = {1000.0, 1001.0, 1002.0};
    cnn::Tensor r = sm.forward(h);
    for (double v : r.data) assert(isfinite(v));
    assert(close(r.data[2], 0.665241, 1e-6));

    // large negative inputs: no divide by zero
    cnn::Tensor m(1, 3, 1, 1);
    m.data = {-1000.0, -1001.0, -1002.0};
    cnn::Tensor t = sm.forward(m);
    for (double v : t.data) assert(isfinite(v));
    assert(close(t.data[0], 0.665241, 1e-6));

    // adding the same number to every score changes nothing
    cnn::Tensor x(1, 3, 1, 1), y(1, 3, 1, 1);
    x.data = {0.0, 1.0, 2.0};
    y.data = {50.0, 51.0, 52.0};
    cnn::Tensor px = sm.forward(x), py = sm.forward(y);
    for (size_t i = 0; i < 3; ++i) assert(close(px.data[i], py.data[i], 1e-12));

    // one class only -> probability 1
    cnn::Tensor one(1, 1, 1, 1);
    one.data = {7.0};
    assert(close(sm.forward(one).data[0], 1.0, 1e-12));

    // output() keeps the last result
    sm.forward(a);
    assert(sm.output().size() == 3 && close(sm.output()[0], 0.659001, 1e-6));

    // wrong shape, NaN input and Inf input are rejected
    bool thrown = false;
    try { cnn::Tensor bad(1, 2, 2, 2); sm.forward(bad); }
    catch (const invalid_argument&) { thrown = true; }
    assert(thrown);

    thrown = false;
    try { cnn::Tensor bad(1, 2, 1, 1); bad.data[0] = NAN; sm.forward(bad); }
    catch (const invalid_argument&) { thrown = true; }
    assert(thrown);

    thrown = false;
    try { cnn::Tensor bad(1, 2, 1, 1); bad.data[1] = INFINITY; sm.forward(bad); }
    catch (const invalid_argument&) { thrown = true; }
    assert(thrown);

    // works after the fully connected layer
    // W = [[1,0],[0,1],[1,1]], b = 0, x = [1, 2] -> z = [1, 2, 3]
    cnn::FullyConnected fc(2, 3);
    fc.setWeights({1, 0, 0, 1, 1, 1});
    cnn::Tensor in(1, 2, 1, 1);
    in.data = {1.0, 2.0};
    cnn::Tensor probs = sm.forward(fc.forward(in));
    double s = probs.data[0] + probs.data[1] + probs.data[2];
    assert(close(s, 1.0, 1e-12));
    assert(probs.data[2] > probs.data[1] && probs.data[1] > probs.data[0]);

    cout << "All softmax tests passed\n";
    return 0;
}
