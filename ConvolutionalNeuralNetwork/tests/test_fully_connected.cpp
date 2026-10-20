#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../include/fully_connected.hpp"

using namespace std;
using namespace cnn;
using Vec = vector<double>;

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

// deterministic pseudo-data: v[i] = ((i * mul) % mod) - shift
static Vec pattern(size_t n, size_t mul, size_t mod, double shift) {
    Vec v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<double>((i * mul) % mod) - shift;
    return v;
}

// 3 inputs -> 2 outputs, W = [[1,2,3],[4,5,6]], b = [0.5, -1]
static FullyConnected sampleLayer() {
    FullyConnected fc(3, 2);
    fc.setWeights({1, 2, 3, 4, 5, 6});
    fc.setBias({0.5, -1});
    return fc;
}

// Independent reference with a different loop order and the bias added last.
static Vec reference(const Vec& W, const Vec& b, const Vec& x, size_t N, size_t in, size_t out) {
    Vec z(N * out, 0.0);
    for (size_t o = 0; o < out; ++o)
        for (size_t i = 0; i < in; ++i)
            for (size_t n = 0; n < N; ++n) z[n * out + o] += W[o * in + i] * x[n * in + i];
    for (size_t n = 0; n < N; ++n)
        for (size_t o = 0; o < out; ++o) z[n * out + o] += b[o];
    return z;
}

static void test_hand_computed() {
    FullyConnected fc = sampleLayer();

    // x = [1, 0, -1]: z0 = 1 + 0 - 3 + 0.5 = -1.5 ; z1 = 4 + 0 - 6 - 1 = -3
    const Tensor one = fc.forward(make(1, 3, 1, 1, {1, 0, -1}));
    CHECK(shapeIs(one, 1, 2, 1, 1));
    CHECK(approx(one.data[0], -1.5) && approx(one.data[1], -3.0));

    // batch of two: second sample x = [2, 1, 0]: z0 = 2 + 2 + 0 + 0.5 = 4.5 ; z1 = 8 + 5 + 0 - 1 = 12
    const Tensor two = fc.forward(make(2, 3, 1, 1, {1, 0, -1, 2, 1, 0}));
    CHECK(shapeIs(two, 2, 2, 1, 1));
    CHECK(allApprox(two.data, {-1.5, -3.0, 4.5, 12.0}));

    // single feature: 3 * 2 - 1 = 5
    FullyConnected tiny(1, 1);
    tiny.setWeights({3});
    tiny.setBias({-1});
    CHECK(approx(tiny.forward(make(1, 1, 1, 1, {2})).data[0], 5.0));
}

static void test_special_cases() {
    // identity weights, zero bias: output equals input
    FullyConnected id(3, 3);
    id.setWeights({1, 0, 0, 0, 1, 0, 0, 0, 1});
    const Tensor x = make(2, 3, 1, 1, {7, -8, 9, 0.5, 0.25, -2});
    CHECK(allApprox(id.forward(x).data, x.data));

    // zero weights: every sample gives the bias
    FullyConnected zero(4, 2);
    zero.setBias({3, -4});
    const Tensor z = zero.forward(make(3, 4, 1, 1, pattern(12, 3, 5, 2.0)));
    CHECK(allApprox(z.data, {3, -4, 3, -4, 3, -4}));

    // zero input gives the bias
    FullyConnected fc = sampleLayer();
    CHECK(allApprox(fc.forward(make(1, 3, 1, 1, {0, 0, 0})).data, {0.5, -1.0}));
}

static void test_matches_reference_and_batch_consistency() {
    const size_t N = 5, in = 7, out = 4;
    const Vec W = pattern(in * out, 5, 11, 5.0);
    const Vec b = pattern(out, 3, 7, 3.0);
    const Vec x = pattern(N * in, 7, 13, 6.0);

    FullyConnected fc(in, out);
    fc.setWeights(W);
    fc.setBias(b);
    const Tensor batch = fc.forward(make(N, in, 1, 1, x));
    CHECK(shapeIs(batch, N, out, 1, 1));
    CHECK(allApprox(batch.data, reference(W, b, x, N, in, out), 1e-9));

    // a batch gives the same numbers as the samples one at a time
    for (size_t n = 0; n < N; ++n) {
        const Vec xn(x.begin() + static_cast<ptrdiff_t>(n * in), x.begin() + static_cast<ptrdiff_t>((n + 1) * in));
        const Tensor single = fc.infer(make(1, in, 1, 1, xn));
        for (size_t o = 0; o < out; ++o) CHECK(approx(single.data[o], batch.data[n * out + o], 1e-12));
    }
}

static void test_linearity() {
    const size_t in = 6, out = 3;
    FullyConnected fc(in, out);  // zero bias
    fc.setWeights(pattern(in * out, 4, 9, 4.0));
    const Vec a = pattern(in, 3, 7, 3.0);
    const Vec c = pattern(in, 5, 8, 2.0);
    Vec sum(in), twice(in);
    for (size_t i = 0; i < in; ++i) {
        sum[i] = a[i] + c[i];
        twice[i] = 2.0 * a[i];
    }
    const Tensor fa = fc.infer(make(1, in, 1, 1, a));
    const Tensor fcx = fc.infer(make(1, in, 1, 1, c));
    const Tensor fsum = fc.infer(make(1, in, 1, 1, sum));
    const Tensor ftwice = fc.infer(make(1, in, 1, 1, twice));
    for (size_t o = 0; o < out; ++o) {
        CHECK(approx(fsum.data[o], fa.data[o] + fcx.data[o]));  // additivity
        CHECK(approx(ftwice.data[o], 2.0 * fa.data[o]));        // homogeneity
    }
}

static void test_feature_map_input() {
    // A raw feature map {1,2,2,2} (8 values per sample) equals the flattened {1,8,1,1}.
    FullyConnected fc(8, 3);
    fc.setWeights(pattern(24, 5, 7, 3.0));
    fc.setBias({0.1, 0.2, 0.3});
    const Vec data = pattern(8, 3, 11, 4.0);
    const Tensor a = fc.infer(make(1, 8, 1, 1, data));
    const Tensor b = fc.infer(make(1, 2, 2, 2, data));
    CHECK(shapeIs(b, 1, 3, 1, 1));
    CHECK(a.data == b.data);
}

static void test_parameters_and_accessors() {
    FullyConnected fc = sampleLayer();
    CHECK(fc.inFeatures() == 3 && fc.outFeatures() == 2);
    CHECK(fc.parameterCount() == 3 * 2 + 2);
    CHECK(fc.weights() == Vec({1, 2, 3, 4, 5, 6}));
    CHECK(fc.bias() == Vec({0.5, -1}));
    CHECK(fc.weight(1, 2) == 6.0 && fc.weight(0, 0) == 1.0);
    CHECK(throwsType<out_of_range>([&] { (void)fc.weight(2, 0); }));
    CHECK(throwsType<out_of_range>([&] { (void)fc.weight(0, 3); }));

    FullyConnected big(400, 120);  // a LeNet-style layer: 400*120 + 120 = 48120
    CHECK(big.parameterCount() == 48120);
    const Tensor zero = big.infer(Tensor(2, 400, 1, 1));
    CHECK(shapeIs(zero, 2, 120, 1, 1));

    // a new layer starts at zero
    FullyConnected fresh(2, 2);
    CHECK(fresh.weights() == Vec({0, 0, 0, 0}) && fresh.bias() == Vec({0, 0}));
}

static void test_invalid_construction_and_setters() {
    CHECK(throwsType<invalid_argument>([&] { FullyConnected f(0, 3); }));
    CHECK(throwsType<invalid_argument>([&] { FullyConnected f(3, 0); }));
    CHECK(throwsType<invalid_argument>([&] { FullyConnected f(static_cast<size_t>(-1), 2); }));

    FullyConnected fc = sampleLayer();
    const Vec w0 = fc.weights();
    const Vec b0 = fc.bias();
    CHECK(throwsType<invalid_argument>([&] { fc.setWeights(Vec(5, 1.0)); }));
    CHECK(throwsType<invalid_argument>([&] { fc.setWeights({1, 2, 3, 4, 5, nan("")}); }));
    CHECK(throwsType<invalid_argument>([&] { fc.setBias(Vec(3, 0.0)); }));
    CHECK(throwsType<invalid_argument>([&] { fc.setBias({0.0, numeric_limits<double>::infinity()}); }));
    CHECK(fc.weights() == w0 && fc.bias() == b0);  // a failed setter changes nothing
}

static void test_invalid_inputs() {
    FullyConnected fc = sampleLayer();
    CHECK(throwsType<invalid_argument>([&] { (void)fc.forward(make(1, 4, 1, 1, {1, 2, 3, 4})); }));  // 4 features, need 3
    CHECK(throwsType<invalid_argument>([&] { (void)fc.forward(make(1, 3, 1, 1, {1, nan(""), 3})); }));
    CHECK(throwsType<invalid_argument>([&] {
        (void)fc.forward(make(1, 3, 1, 1, {1, numeric_limits<double>::infinity(), 3}));
    }));

    Tensor broken(1, 3, 1, 1);
    broken.data.pop_back();  // data size does not match shape
    CHECK(throwsType<invalid_argument>([&] { (void)fc.forward(broken); }));
    Tensor rank3(1, 3, 1, 1);
    rank3.shape.pop_back();  // not 4-D
    CHECK(throwsType<invalid_argument>([&] { (void)fc.forward(rank3); }));

    // overflow during the computation is reported, not returned as infinity
    FullyConnected huge(2, 1);
    huge.setWeights({1e308, 1e308});
    CHECK(throwsType<runtime_error>([&] { (void)huge.forward(make(1, 2, 1, 1, {10, 10})); }));
}

static void test_cache() {
    FullyConnected fc = sampleLayer();
    CHECK(!fc.hasCachedInput());
    CHECK(throwsType<logic_error>([&] { (void)fc.cachedInput(); }));

    Tensor x = make(1, 3, 1, 1, {1, 2, 3});
    (void)fc.forward(x);
    CHECK(fc.hasCachedInput());
    CHECK(fc.cachedInput().data == Vec({1, 2, 3}));
    CHECK(shapeIs(fc.cachedInput(), 1, 3, 1, 1));

    x.data[0] = 99.0;  // the cache is a copy, not a reference
    CHECK(fc.cachedInput().data[0] == 1.0);

    CHECK(throwsType<invalid_argument>([&] { (void)fc.forward(make(1, 4, 1, 1, {1, 2, 3, 4})); }));
    CHECK(fc.cachedInput().data == Vec({1, 2, 3}));  // a failed forward keeps the old cache

    (void)fc.infer(make(1, 3, 1, 1, {7, 8, 9}));      // inference does not touch the cache
    CHECK(fc.cachedInput().data == Vec({1, 2, 3}));

    (void)fc.forward(make(2, 3, 1, 1, {4, 5, 6, 7, 8, 9}));  // a new forward replaces it
    CHECK(fc.cachedInput().data == Vec({4, 5, 6, 7, 8, 9}));
    CHECK(shapeIs(fc.cachedInput(), 2, 3, 1, 1));

    fc.clearCache();
    CHECK(!fc.hasCachedInput());
}

int main() {
    test_hand_computed();
    test_special_cases();
    test_matches_reference_and_batch_consistency();
    test_linearity();
    test_feature_map_input();
    test_parameters_and_accessors();
    test_invalid_construction_and_setters();
    test_invalid_inputs();
    test_cache();
    if (failures == 0) cout << "test_fully_connected: all passed\n";
    return failures == 0 ? 0 : 1;
}