#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "fully_connected_trainable.hpp"

using namespace std;
using namespace cnn;
using Vec = vector<double>;

static_assert(is_base_of<FullyConnected, FullyConnectedTrainable>::value,
              "FullyConnectedTrainable must derive from FullyConnected");
static_assert(!is_default_constructible<FullyConnectedTrainable>::value,
              "a layer needs its sizes: no default constructor");

static int failures = 0;
static void check_impl(bool ok, const char* expr, const char* file, int line) {
    if (!ok) {
        cerr << "FAIL " << file << ":" << line << "  " << expr << "\n";
        ++failures;
    }
}
#define CHECK(cond) check_impl((cond), #cond, __FILE__, __LINE__)

template <typename E, typename F>
static bool throwsType(F&& f) {
    try { f(); } catch (const E&) { return true; } catch (...) { return false; }
    return false;
}

static bool approx(double a, double b, double eps = 1e-9) { return fabs(a - b) <= eps; }

static bool allApprox(const Vec& a, const Vec& b, double eps = 1e-9) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!approx(a[i], b[i], eps)) return false;
    }
    return true;
}

static bool shapeIs(const Tensor& t, size_t n, size_t c, size_t h, size_t w) {
    return t.shape == vector<size_t>({n, c, h, w});
}

static Tensor make(size_t n, size_t c, size_t h, size_t w, const Vec& v) {
    Tensor t(n, c, h, w);
    t.data = v;
    return t;
}

// deterministic pseudo-data: v[i] = (((i * mul) % mod) - shift) * scale
static Vec pattern(size_t n, size_t mul, size_t mod, double shift, double scale = 1.0) {
    Vec v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (static_cast<double>((i * mul) % mod) - shift) * scale;
    return v;
}

// 3 inputs -> 2 outputs, W = [[1,2,3],[4,5,6]], b = [0.5, -1]
static FullyConnectedTrainable sampleLayer() {
    FullyConnectedTrainable fc(3, 2);
    fc.setWeights({1, 2, 3, 4, 5, 6});
    fc.setBias({0.5, -1});
    return fc;
}

// Loss used for the numerical check: L = 0.5 * sum of z^2, so dL/dz = z.
// It takes the BASE class, so it also shows that a derived object can be used as its base.
static double halfSquaredLoss(const FullyConnected& fc, const Tensor& x) {
    const Tensor z = fc.infer(x);
    double s = 0.0;
    for (double v : z.data) s += 0.5 * v * v;
    return s;
}

// ---------------------------------------------------------------------------
static void test_inheritance() {
    FullyConnectedTrainable t = sampleLayer();
    FullyConnected& base = t;  // a derived object can be used where the base class is expected
    CHECK(base.inFeatures() == 3 && base.outFeatures() == 2);
    CHECK(base.parameterCount() == 8);

    const Tensor x = make(1, 3, 1, 1, {1, 0, -1});
    const Tensor viaBase = base.infer(x);
    const Tensor viaDerived = t.infer(x);
    CHECK(viaBase.data == viaDerived.data);
    CHECK(allApprox(viaBase.data, {-1.5, -3.0}));

    (void)t.forward(x);  // forward() is inherited and saves the input for backward()
    CHECK(t.hasCachedInput() && base.hasCachedInput());
    CHECK(t.gradWeights().size() == 6 && t.gradBias().size() == 2);

    CHECK(throwsType<invalid_argument>([&] { FullyConnectedTrainable bad(0, 3); }));  // base class checks still apply
}

static void test_hand_computed_single_sample() {
    FullyConnectedTrainable fc = sampleLayer();
    (void)fc.forward(make(1, 3, 1, 1, {1, 0, -1}));
    const Tensor dx = fc.backward(make(1, 2, 1, 1, {1, 2}));

    // dW = delta (outer) x = [[1,0,-1],[2,0,-2]], db = delta, dx = W^T delta = [1+8, 2+10, 3+12]
    CHECK(allApprox(fc.gradWeights(), {1, 0, -1, 2, 0, -2}));
    CHECK(allApprox(fc.gradBias(), {1, 2}));
    CHECK(shapeIs(dx, 1, 3, 1, 1));
    CHECK(allApprox(dx.data, {9, 12, 15}));
}

static void test_hand_computed_batch() {
    FullyConnectedTrainable fc = sampleLayer();
    (void)fc.forward(make(2, 3, 1, 1, {1, 0, -1, 2, 1, 0}));
    const Tensor dx = fc.backward(make(2, 2, 1, 1, {1, 2, 0.5, -1}));

    // row o=0: 1*[1,0,-1] + 0.5*[2,1,0] = [2,0.5,-1]; row o=1: 2*[1,0,-1] - 1*[2,1,0] = [0,-1,-2]
    CHECK(allApprox(fc.gradWeights(), {2, 0.5, -1, 0, -1, -2}));
    CHECK(allApprox(fc.gradBias(), {1.5, 1}));
    CHECK(shapeIs(dx, 2, 3, 1, 1));
    // sample 0: W^T [1,2] = [9,12,15]; sample 1: W^T [0.5,-1] = [-3.5,-4,-4.5]
    CHECK(allApprox(dx.data, {9, 12, 15, -3.5, -4, -4.5}));
}

// Central differences: (L(p + eps) - L(p - eps)) / (2 eps) must match the analytic gradient.
static void test_numerical_gradient_check() {
    const size_t N = 3, in = 5, out = 4;
    FullyConnectedTrainable fc(in, out);
    fc.setWeights(pattern(in * out, 7, 11, 5.0, 0.1));
    fc.setBias(pattern(out, 3, 5, 2.0, 0.1));
    const Tensor x = make(N, in, 1, 1, pattern(N * in, 5, 13, 6.0, 0.1));

    const Tensor z = fc.forward(x);
    const Tensor dx = fc.backward(z);  // dL/dz = z for the half squared loss
    const double eps = 1e-5, tol = 1e-6;

    for (size_t k = 0; k < fc.weights().size(); ++k) {
        FullyConnectedTrainable f = fc;
        Vec w = f.weights();
        w[k] += eps;
        f.setWeights(w);
        const double up = halfSquaredLoss(f, x);
        w[k] -= 2 * eps;
        f.setWeights(w);
        const double down = halfSquaredLoss(f, x);
        CHECK(approx((up - down) / (2 * eps), fc.gradWeights()[k], tol));
    }
    for (size_t k = 0; k < fc.bias().size(); ++k) {
        FullyConnectedTrainable f = fc;
        Vec b = f.bias();
        b[k] += eps;
        f.setBias(b);
        const double up = halfSquaredLoss(f, x);
        b[k] -= 2 * eps;
        f.setBias(b);
        const double down = halfSquaredLoss(f, x);
        CHECK(approx((up - down) / (2 * eps), fc.gradBias()[k], tol));
    }
    for (size_t k = 0; k < x.data.size(); ++k) {
        Tensor xp = x, xm = x;
        xp.data[k] += eps;
        xm.data[k] -= eps;
        CHECK(approx((halfSquaredLoss(fc, xp) - halfSquaredLoss(fc, xm)) / (2 * eps), dx.data[k], tol));
    }
}

static void test_sum_over_batch_not_mean() {
    FullyConnectedTrainable one = sampleLayer();
    (void)one.forward(make(1, 3, 1, 1, {1, 2, 3}));
    const Tensor dx1 = one.backward(make(1, 2, 1, 1, {0.5, -2}));

    FullyConnectedTrainable two = sampleLayer();
    (void)two.forward(make(2, 3, 1, 1, {1, 2, 3, 1, 2, 3}));  // same sample twice
    const Tensor dx2 = two.backward(make(2, 2, 1, 1, {0.5, -2, 0.5, -2}));

    for (size_t i = 0; i < one.gradWeights().size(); ++i) {
        CHECK(approx(two.gradWeights()[i], 2.0 * one.gradWeights()[i]));  // summed, not averaged
    }
    for (size_t o = 0; o < one.gradBias().size(); ++o) CHECK(approx(two.gradBias()[o], 2.0 * one.gradBias()[o]));
    for (size_t i = 0; i < 3; ++i) {
        CHECK(approx(dx2.data[i], dx1.data[i]));      // each sample's dx is independent of the batch
        CHECK(approx(dx2.data[3 + i], dx1.data[i]));
    }
}

static void test_input_shape_is_kept() {
    // A feature map {1,2,2,2} (8 values) gives dx in the same shape, and the same numbers as {1,8,1,1}.
    FullyConnectedTrainable a(8, 3), b(8, 3);
    const Vec w = pattern(24, 5, 7, 3.0, 0.5);
    a.setWeights(w);
    b.setWeights(w);
    const Vec data = pattern(8, 3, 11, 4.0);
    (void)a.forward(make(1, 2, 2, 2, data));
    (void)b.forward(make(1, 8, 1, 1, data));
    const Tensor dxa = a.backward(make(1, 3, 1, 1, {1, -1, 2}));
    const Tensor dxb = b.backward(make(1, 3, 1, 1, {1, -1, 2}));
    CHECK(shapeIs(dxa, 1, 2, 2, 2));
    CHECK(shapeIs(dxb, 1, 8, 1, 1));
    CHECK(dxa.data == dxb.data);
    CHECK(a.gradWeights() == b.gradWeights());

    // grad_output may also arrive in any 4-D shape with the right number of values per sample
    (void)b.forward(make(1, 8, 1, 1, data));
    const Tensor dxc = b.backward(make(1, 1, 1, 3, {1, -1, 2}));
    CHECK(dxc.data == dxb.data);
}

static void test_gradient_storage() {
    FullyConnectedTrainable fc(2, 2);
    CHECK(fc.gradWeights() == Vec({0, 0, 0, 0}) && fc.gradBias() == Vec({0, 0}));  // zero before any backward

    fc.setWeights({1, 0, 0, 1});
    (void)fc.forward(make(1, 2, 1, 1, {3, 4}));
    (void)fc.backward(make(1, 2, 1, 1, {1, 1}));
    CHECK(fc.gradWeights() == Vec({3, 4, 3, 4}) && fc.gradBias() == Vec({1, 1}));

    (void)fc.forward(make(1, 2, 1, 1, {1, 1}));  // a new backward replaces, it does not add
    (void)fc.backward(make(1, 2, 1, 1, {2, 0}));
    CHECK(fc.gradWeights() == Vec({2, 2, 0, 0}) && fc.gradBias() == Vec({2, 0}));

    fc.zeroGrad();
    CHECK(fc.gradWeights() == Vec({0, 0, 0, 0}) && fc.gradBias() == Vec({0, 0}));
}

static void test_backward_errors() {
    FullyConnectedTrainable fc = sampleLayer();
    CHECK(throwsType<logic_error>([&] { (void)fc.backward(make(1, 2, 1, 1, {1, 2})); }));  // no forward yet

    (void)fc.forward(make(1, 3, 1, 1, {1, 0, -1}));
    (void)fc.backward(make(1, 2, 1, 1, {1, 2}));
    const Vec w0 = fc.gradWeights(), b0 = fc.gradBias();

    CHECK(throwsType<invalid_argument>([&] { (void)fc.backward(make(2, 2, 1, 1, {1, 2, 3, 4})); }));  // batch differs
    CHECK(throwsType<invalid_argument>([&] { (void)fc.backward(make(1, 3, 1, 1, {1, 2, 3})); }));     // wrong width
    CHECK(throwsType<invalid_argument>([&] { (void)fc.backward(make(1, 2, 1, 1, {1, nan("")})); }));
    CHECK(throwsType<invalid_argument>([&] {
        (void)fc.backward(make(1, 2, 1, 1, {numeric_limits<double>::infinity(), 0}));
    }));
    Tensor broken(1, 2, 1, 1);
    broken.data.pop_back();
    CHECK(throwsType<invalid_argument>([&] { (void)fc.backward(broken); }));
    Tensor rank3(1, 2, 1, 1);
    rank3.shape.pop_back();
    CHECK(throwsType<invalid_argument>([&] { (void)fc.backward(rank3); }));
    CHECK(fc.gradWeights() == w0 && fc.gradBias() == b0);  // failed calls change nothing

    fc.clearCache();  // inherited from the base class
    CHECK(throwsType<logic_error>([&] { (void)fc.backward(make(1, 2, 1, 1, {1, 2})); }));  // cache cleared

    // overflow: the forward pass is fine (zero input), the backward dx = 1e308 * 10 is not
    FullyConnectedTrainable huge(2, 1);
    huge.setWeights({1e308, 1e308});
    (void)huge.forward(make(1, 2, 1, 1, {0, 0}));
    CHECK(throwsType<runtime_error>([&] { (void)huge.backward(make(1, 1, 1, 1, {10})); }));
    CHECK(huge.gradWeights() == Vec({0, 0}));
}

static void test_apply_update() {
    FullyConnectedTrainable fc = sampleLayer();
    fc.applyUpdate({0.5, 0.5, 0.5, -1, -1, -1}, {1, -1});
    CHECK(allApprox(fc.weights(), {1.5, 2.5, 3.5, 3, 4, 5}));
    CHECK(allApprox(fc.bias(), {1.5, -2}));

    const Vec w0 = fc.weights(), b0 = fc.bias();
    CHECK(throwsType<invalid_argument>([&] { fc.applyUpdate(Vec(5, 0.0), {0, 0}); }));
    CHECK(throwsType<invalid_argument>([&] { fc.applyUpdate(Vec(6, 0.0), {0}); }));
    CHECK(throwsType<invalid_argument>([&] { fc.applyUpdate({0, 0, 0, 0, 0, nan("")}, {0, 0}); }));
    CHECK(fc.weights() == w0 && fc.bias() == b0);  // unchanged after failures

    FullyConnectedTrainable huge(2, 1);
    huge.setWeights({1e308, 1e308});
    CHECK(throwsType<runtime_error>([&] { huge.applyUpdate({1e308, 1e308}, {0}); }));  // would overflow
    CHECK(huge.weights() == Vec({1e308, 1e308}));

    // the bias must not change when the weight update fails (no half-applied update)
    FullyConnectedTrainable half(2, 1);
    half.setWeights({1e308, 0});
    half.setBias({5});
    CHECK(throwsType<runtime_error>([&] { half.applyUpdate({1e308, 0}, {1}); }));
    CHECK(half.bias() == Vec({5}));
}

// One gradient-descent step must lower the loss; twenty steps must nearly remove it.
static void test_training_steps_reduce_loss() {
    FullyConnectedTrainable fc = sampleLayer();
    const Tensor x = make(2, 3, 1, 1, {1, 0, -1, 2, 1, 0});
    const double first = halfSquaredLoss(fc, x);
    double previous = first;
    // Stable for this data: the loss Hessian has largest eigenvalue about 7.85, so lr must stay below 2/7.85.
    const double lr = 0.2;
    for (int step = 0; step < 20; ++step) {
        const Tensor z = fc.forward(x);
        (void)fc.backward(z);  // dL/dz = z
        Vec dw = fc.gradWeights(), db = fc.gradBias();
        for (double& v : dw) v *= -lr;
        for (double& v : db) v *= -lr;
        fc.applyUpdate(dw, db);
        const double now = halfSquaredLoss(fc, x);
        CHECK(now < previous);
        previous = now;
    }
    CHECK(previous < 0.01 * first);
}

int main() {
    test_inheritance();
    test_hand_computed_single_sample();
    test_hand_computed_batch();
    test_numerical_gradient_check();
    test_sum_over_batch_not_mean();
    test_input_shape_is_kept();
    test_gradient_storage();
    test_backward_errors();
    test_apply_update();
    test_training_steps_reduce_loss();
    if (failures == 0) cout << "test_fully_connected_trainable: all passed\n";
    return failures == 0 ? 0 : 1;
}