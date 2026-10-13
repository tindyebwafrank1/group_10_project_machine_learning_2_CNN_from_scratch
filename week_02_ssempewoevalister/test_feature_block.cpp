// Plain-assert tests, no external dependencies.
#include <cmath>
#include <iostream>
#include "cnn_library/feature_block.hpp"

using namespace CNN;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { ++failures; \
    std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n"; } } while (0)
#define CHECK_NEAR(a, b) CHECK(std::fabs((a) - (b)) < 1e-9)
#define CHECK_THROWS(expr) do { bool t = false; try { expr; } catch (const std::exception&) { t = true; } \
    CHECK(t); } while (0)

static Tensor ramp_4x4() { // values -7 .. 8
    Tensor t(1, 1, 4, 4);
    for (size_t i = 0; i < 16; ++i) t.data[i] = double(i) - 7.0;
    return t;
}
static Tensor ones_kernel(size_t o, size_t c, size_t k) {
    Tensor t(o, c, k, k);
    for (double& v : t.data) v = 1.0;
    return t;
}

void test_relu_maxpool_values() {
    FeatureBlock b(1, 1, 1, 1, 0);
    b.set_kernel(ones_kernel(1, 1, 1));
    Tensor out = b.forward(ramp_4x4());
    CHECK(out.shape[2] == 2 && out.shape[3] == 2);
    CHECK_NEAR(out(0,0,0,0), 0.0); CHECK_NEAR(out(0,0,0,1), 0.0);
    CHECK_NEAR(out(0,0,1,0), 6.0); CHECK_NEAR(out(0,0,1,1), 8.0);
}

void test_avgpool_values() {
    FeatureBlock b(1, 1, 1, 1, 0);
    b.set_kernel(ones_kernel(1, 1, 1));
    b.pool = Pooling(2, 2, Pooling::Mode::Average);
    Tensor out = b.forward(ramp_4x4());
    CHECK_NEAR(out(0,0,1,0), 3.5); CHECK_NEAR(out(0,0,1,1), 5.5);
}

void test_same_padding() {
    FeatureBlock b(1, 1, 3, 1, 1);
    b.set_kernel(ones_kernel(1, 1, 3));
    b.pool = Pooling(2, 2, Pooling::Mode::Average);
    Tensor in(1, 1, 4, 4);
    for (double& v : in.data) v = 1.0;
    Tensor out = b.forward(in); // conv: corners 4, edges 6, interior 9; avg of top-left block = 6.25
    CHECK(out.shape[2] == 2 && out.shape[3] == 2);
    CHECK_NEAR(out(0,0,0,0), 6.25);
}

void test_multichannel_multifilter() {
    FeatureBlock b(2, 2, 1, 1, 0);
    Tensor k(2, 2, 1, 1);
    k(0,0,0,0) = 1; k(0,1,0,0) = 1;   // filter 0 = ch0 + ch1
    k(1,0,0,0) = 1; k(1,1,0,0) = -1;  // filter 1 = ch0 - ch1
    b.set_kernel(k);
    Tensor in(1, 2, 2, 2);
    for (size_t h = 0; h < 2; ++h) for (size_t w = 0; w < 2; ++w) { in(0,0,h,w) = 2; in(0,1,h,w) = 1; }
    Tensor out = b.forward(in);
    CHECK(out.shape[1] == 2 && out.shape[2] == 1 && out.shape[3] == 1);
    CHECK_NEAR(out(0,0,0,0), 3.0); CHECK_NEAR(out(0,1,0,0), 1.0);
}

void test_batch() {
    FeatureBlock b(1, 3, 3, 1, 0);
    Tensor in(5, 1, 8, 8);
    Tensor out = b.forward(in);
    CHECK(out.shape[0] == 5 && out.shape[1] == 3 && out.shape[2] == 3 && out.shape[3] == 3);
}

void test_activations() {
    Tensor in(1, 1, 4, 4);
    for (double& v : in.data) v = 5.0;
    Tensor zeros(1, 1, 1, 1);
    FeatureBlock s(1, 1, 1, 1, 0, Activation::Sigmoid);
    s.set_kernel(zeros);
    CHECK_NEAR(s.forward(in)(0,0,0,0), 0.5);
    FeatureBlock t(1, 1, 1, 1, 0, Activation::Tanh);
    t.set_kernel(zeros);
    CHECK_NEAR(t.forward(in)(0,0,0,0), 0.0);
}

void test_shape_formula_15_combos() {
    struct C { size_t H, W, P, S, K; };
    const C combos[15] = {
        {8,8,0,1,3},{8,8,1,1,3},{8,8,0,2,3},{10,10,2,1,5},{10,12,1,2,3},
        {7,7,0,1,3},{7,9,1,1,1},{16,16,0,4,4},{16,16,2,2,5},{5,5,0,1,1},
        {6,6,3,3,3},{9,9,0,3,3},{12,12,1,1,5},{14,10,0,2,3},{20,20,1,3,4}};
    for (const C& c : combos) {
        FeatureBlock b(1, 2, c.K, c.S, c.P);
        size_t ch = (c.H + 2*c.P - c.K) / c.S + 1, cw = (c.W + 2*c.P - c.K) / c.S + 1;
        size_t eh = (ch - 2) / 2 + 1, ew = (cw - 2) / 2 + 1;
        Tensor out = b.forward(Tensor(1, 1, c.H, c.W));
        CHECK(out.shape[1] == 2 && out.shape[2] == eh && out.shape[3] == ew);
        auto s = b.output_shape(c.H, c.W);
        CHECK(s[1] == eh && s[2] == ew);
    }
}

void test_seed_determinism() {
    FeatureBlock a(2, 3, 3, 1, 0, Activation::ReLU, 7), b(2, 3, 3, 1, 0, Activation::ReLU, 7),
                 c(2, 3, 3, 1, 0, Activation::ReLU, 8);
    CHECK(a.kernel_weights().data == b.kernel_weights().data);
    CHECK(a.kernel_weights().data != c.kernel_weights().data);
}

void test_extractor_builder() {
    FeatureExtractor fx(1);
    fx.add_block(4, 3, 1, 1).add_block(8, 3, 1, 1).add_block(16, 3, 1, 0, Activation::Tanh);
    CHECK(fx.num_blocks() == 3);
    auto s = fx.output_shape(28, 28); // 28 -> 14 -> 7 -> (7-3+1=5)->2
    CHECK(s[0] == 16 && s[1] == 2 && s[2] == 2);
    Tensor out = fx.forward(Tensor(2, 1, 28, 28));
    CHECK(out.shape[0] == 2 && out.shape[1] == 16 && out.shape[2] == 2 && out.shape[3] == 2);
    CHECK(fx.block(1).in_channels() == 4);
}

void test_invalid_inputs() {
    CHECK_THROWS(FeatureBlock(0, 1, 3, 1, 0));
    CHECK_THROWS(FeatureBlock(1, 0, 3, 1, 0));
    CHECK_THROWS(FeatureBlock(1, 1, 0, 1, 0));
    CHECK_THROWS(FeatureBlock(1, 1, 3, 0, 0));
    FeatureBlock b(2, 2, 3, 1, 0);
    CHECK_THROWS(b.forward(Tensor(1, 3, 8, 8)));        // wrong channels
    CHECK_THROWS(b.forward(Tensor(1, 2, 2, 2)));        // kernel > input
    CHECK_THROWS(b.forward(Tensor(1, 2, 3, 3)));        // conv gives 1x1, smaller than pool window
    CHECK_THROWS(b.set_kernel(Tensor(1, 2, 3, 3)));     // wrong kernel shape
    FeatureBlock c(1, 1, 3, 1, 0);
    CHECK_NEAR(c.forward(Tensor(1, 1, 4, 4)).shape[2], 1.0); // 4x4 -> conv 2x2 -> pool 1x1 is valid
    CHECK_THROWS(FeatureExtractor(0));
    FeatureExtractor empty(1);
    CHECK_THROWS(empty.forward(Tensor(1, 1, 8, 8)));
}

int main() {
    test_relu_maxpool_values(); test_avgpool_values(); test_same_padding();
    test_multichannel_multifilter(); test_batch(); test_activations();
    test_shape_formula_15_combos(); test_seed_determinism();
    test_extractor_builder(); test_invalid_inputs();
    if (failures == 0) { std::cout << "All feature block tests passed.\n"; return 0; }
    std::cerr << failures << " check(s) failed.\n";
    return 1;
}
