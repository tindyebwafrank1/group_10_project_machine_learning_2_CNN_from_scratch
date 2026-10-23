// Unit tests for the ReLU, Sigmoid and Tanh modules (forward and backward).

#include <cassert>
#include <cmath>
#include <iostream>
#include "activations.hpp"

// True if a and b differ by less than tol (doubles should never be compared with ==).
static bool close(double a, double b, double tol = 1e-12) {
    return std::fabs(a - b) < tol;
}

int main() {
    // Test input covering extreme negatives, zero, and extreme positives.
    Tensor t(1, 1, 1, 7);
    double xs[7] = {-1000, -5, -1, 0, 1, 5, 1000};
    for (int i = 0; i < 7; ++i) t.data[i] = xs[i];

    act::Sigmoid s;
    act::Tanh th;
    Tensor ys = s.forward(t);
    Tensor yt = th.forward(t);

    for (int i = 0; i < 7; ++i) {
        // Stability: no inf or NaN, even for inputs of +/-1000.
        assert(std::isfinite(ys.data[i]) && std::isfinite(yt.data[i]));
        // Range: sigmoid in [0,1], tanh in [-1,1].
        assert(ys.data[i] >= 0.0 && ys.data[i] <= 1.0);
        assert(yt.data[i] >= -1.0 && yt.data[i] <= 1.0);
    }

    // Known values at zero.
    assert(close(ys.data[3], 0.5)); // sigmoid(0) = 0.5
    assert(close(yt.data[3], 0.0)); // tanh(0) = 0

    // Symmetry properties.
    assert(close(ys.data[4] + ys.data[2], 1.0)); // sigmoid(x) + sigmoid(-x) = 1
    assert(close(yt.data[4], -yt.data[2])); // tanh is an odd function

    // Reference values at x = 1.
    assert(close(ys.data[4], 0.7310585786300049, 1e-12));
    assert(close(yt.data[4], 0.7615941559557649, 1e-12));

    // The output keeps the input's shape, and the input is not modified.
    assert(ys.shape == t.shape && yt.shape == t.shape);
    assert(t.data[0] == -1000);

    // Gradient check: compare backward() with a numerical derivative
    // (f(x+h) - f(x-h)) / 2h at x = 0.7.
    Tensor one(1, 1, 1, 1);
    one.data[0] = 0.7;
    Tensor g(1, 1, 1, 1);
    g.data[0] = 1.0; // upstream gradient of 1
    double h = 1e-6;
    double num_s = (act::sigmoid(0.7 + h) - act::sigmoid(0.7 - h)) / (2 * h);
    double num_t = (std::tanh(0.7 + h) - std::tanh(0.7 - h)) / (2 * h);
    assert(close(s.backward(s.forward(one), g).data[0], num_s, 1e-8));
    assert(close(th.backward(th.forward(one), g).data[0], num_t, 1e-8));

    // ---------- Week 4: ReLU and gradient checks ----------
    act::ReLU re;
    Tensor rx(1, 1, 1, 4);
    double rv[4] = {-2.0, 0.0, 0.5, 3.0};
    for (int i = 0; i < 4; ++i) rx.data[i] = rv[i];
    Tensor ry = re.forward(rx);
    assert(ry.data[0] == 0.0 && ry.data[1] == 0.0 && ry.data[2] == 0.5 && ry.data[3] == 3.0);
    assert(ry.shape == rx.shape && rx.data[0] == -2.0);

    // ReLU backward: gradient passes only where x > 0
    Tensor rg(1, 1, 1, 4);
    rg.fill(5.0);
    Tensor rb = re.backward(ry, rg);
    assert(rb.data[0] == 0.0 && rb.data[1] == 0.0 && rb.data[2] == 5.0 && rb.data[3] == 5.0);
    assert(rg.data[0] == 5.0); // grad_out not modified

    // Sigmoid'(0) = 0.25, Tanh'(0) = 1
    Tensor z(1, 1, 1, 1);
    z.data[0] = 0.0;
    assert(close(s.backward(s.forward(z), g).data[0], 0.25));
    assert(close(th.backward(th.forward(z), g).data[0], 1.0));

    // numerical gradient check at several points (away from ReLU's kink at 0)
    double pts[6] = {-2.5, -0.7, -0.1, 0.3, 1.4, 3.0};
    Tensor px(1, 1, 1, 6);
    for (int i = 0; i < 6; ++i) px.data[i] = pts[i];
    Tensor up(1, 1, 1, 6);
    up.fill(1.0);
    Tensor gr = re.backward(re.forward(px), up);
    Tensor gs = s.backward(s.forward(px), up);
    Tensor gt = th.backward(th.forward(px), up);
    for (int i = 0; i < 6; ++i) {
        double x = pts[i];
        double nr = (std::max(0.0, x + h) - std::max(0.0, x - h)) / (2 * h);
        double ns = (act::sigmoid(x + h) - act::sigmoid(x - h)) / (2 * h);
        double nt = (std::tanh(x + h) - std::tanh(x - h)) / (2 * h);
        assert(close(gr.data[i], nr, 1e-6));
        assert(close(gs.data[i], ns, 1e-8));
        assert(close(gt.data[i], nt, 1e-8));
    }

    // upstream gradient is multiplied (chain rule)
    Tensor two(1, 1, 1, 1);
    two.data[0] = 2.0;
    assert(close(s.backward(s.forward(z), two).data[0], 0.5));

    // size mismatch is rejected
    bool thrown = false;
    try { re.backward(Tensor(1, 1, 1, 2), Tensor(1, 1, 1, 3)); }
    catch (const std::invalid_argument&) { thrown = true; }
    assert(thrown);

    std::cout << "All activation tests passed\n";
    return 0;
}
