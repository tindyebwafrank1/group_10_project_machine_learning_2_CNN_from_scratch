// Plain-assert tests, no external dependencies.
#include <cmath>
#include <iostream>
#include <limits>
#include "cnn_library/optimizer.hpp"

using namespace CNN;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { ++failures; \
    std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n"; } } while (0)
#define CHECK_NEAR(a, b) CHECK(std::fabs((a) - (b)) < 1e-9)
#define CHECK_THROWS(expr) do { bool t = false; try { expr; } catch (const std::exception&) { t = true; } \
    CHECK(t); } while (0)

void test_plain_step() {
    SGD opt(0.1);
    std::vector<double> w = {1.0, -2.0, 0.0};
    opt.step(w, {0.5, -1.0, 2.0});
    CHECK_NEAR(w[0], 0.95); CHECK_NEAR(w[1], -1.9); CHECK_NEAR(w[2], -0.2);
}

void test_zero_gradient_no_change() {
    SGD opt(0.5);
    std::vector<double> w = {3.0, 4.0};
    opt.step(w, {0.0, 0.0});
    CHECK_NEAR(w[0], 3.0); CHECK_NEAR(w[1], 4.0);
}

void test_grad_scale() {
    SGD opt(0.1);
    std::vector<double> w = {1.0};
    opt.step(w, {8.0}, 1.0 / 4.0); // summed gradient over a batch of 4 -> mean 2.0
    CHECK_NEAR(w[0], 0.8);
}

void test_weight_decay() {
    SGD opt(0.1, 0.0, 0.5);
    std::vector<double> w = {2.0};
    opt.step(w, {1.0}); // g_eff = 1 + 0.5*2 = 2 -> w = 2 - 0.2
    CHECK_NEAR(w[0], 1.8);
}

void test_momentum() {
    SGD opt(0.1, 0.9);
    std::vector<double> w = {1.0};
    opt.step(w, {0.5}); // v = -0.05, w = 0.95
    CHECK_NEAR(w[0], 0.95);
    opt.step(w, {0.5}); // v = 0.9*-0.05 - 0.05 = -0.095, w = 0.855
    CHECK_NEAR(w[0], 0.855);
    opt.reset_state();
    std::vector<double> w2 = {1.0};
    opt.step(w2, {0.5});
    CHECK_NEAR(w2[0], 0.95);
}

void test_separate_velocity_per_parameter() {
    SGD opt(0.1, 0.9);
    std::vector<double> a = {1.0}, b = {1.0};
    opt.step(a, {0.5}); opt.step(a, {0.5});
    opt.step(b, {0.5}); // b's first step must not inherit a's momentum
    CHECK_NEAR(a[0], 0.855); CHECK_NEAR(b[0], 0.95);
}

void test_tensor_overload() {
    SGD opt(0.1);
    Tensor w(1, 2, 2, 2), g(1, 2, 2, 2);
    for (size_t i = 0; i < w.data.size(); ++i) { w.data[i] = 1.0; g.data[i] = double(i); }
    opt.step(w, g);
    CHECK_NEAR(w.data[0], 1.0); CHECK_NEAR(w.data[5], 0.5); CHECK_NEAR(w.data[7], 0.3);
    zero_gradients(g);
    for (double v : g.data) CHECK_NEAR(v, 0.0);
}

void test_convergence_quadratic() {
    // minimise (w - 3)^2, gradient 2(w - 3)
    SGD opt(0.1);
    std::vector<double> w = {-5.0};
    for (int i = 0; i < 200; ++i) opt.step(w, {2.0 * (w[0] - 3.0)});
    CHECK(std::fabs(w[0] - 3.0) < 1e-6);
    SGD mom(0.05, 0.9);
    std::vector<double> w2 = {-5.0};
    for (int i = 0; i < 300; ++i) mom.step(w2, {2.0 * (w2[0] - 3.0)});
    CHECK(std::fabs(w2[0] - 3.0) < 1e-4);
}

void test_lr_schedule() {
    SGD opt(0.1);
    opt.decay_learning_rate(0.5);
    CHECK_NEAR(opt.learning_rate(), 0.05);
    opt.set_learning_rate(0.2);
    CHECK_NEAR(opt.learning_rate(), 0.2);
}

void test_invalid_inputs() {
    CHECK_THROWS(SGD(0.0));
    CHECK_THROWS(SGD(-0.1));
    CHECK_THROWS(SGD(std::numeric_limits<double>::infinity()));
    CHECK_THROWS(SGD(0.1, 1.0));
    CHECK_THROWS(SGD(0.1, -0.1));
    CHECK_THROWS(SGD(0.1, 0.0, -1.0));
    SGD opt(0.1);
    CHECK_THROWS(opt.set_learning_rate(0.0));
    CHECK_THROWS(opt.decay_learning_rate(0.0));
    CHECK_THROWS(opt.decay_learning_rate(1.5));
    std::vector<double> w = {1.0, 2.0};
    CHECK_THROWS(opt.step(w, {1.0}));
    CHECK_THROWS(opt.step(w, {1.0, std::numeric_limits<double>::quiet_NaN()}));
    CHECK_THROWS(opt.step(w, {1.0, std::numeric_limits<double>::infinity()}));
    CHECK_NEAR(w[0], 1.0); CHECK_NEAR(w[1], 2.0); // unchanged after failed steps
    Tensor a(1, 1, 2, 2), b(1, 1, 3, 3);
    CHECK_THROWS(opt.step(a, b));
}

void test_empty_params() {
    SGD opt(0.1);
    std::vector<double> w;
    opt.step(w, {}); // no-op, must not crash
    CHECK(w.empty());
}

int main() {
    test_plain_step(); test_zero_gradient_no_change(); test_grad_scale();
    test_weight_decay(); test_momentum(); test_separate_velocity_per_parameter();
    test_tensor_overload(); test_convergence_quadratic(); test_lr_schedule();
    test_invalid_inputs(); test_empty_params();
    if (failures == 0) { std::cout << "All optimizer tests passed.\n"; return 0; }
    std::cerr << failures << " check(s) failed.\n";
    return 1;
}
