#include <cmath>
#include <iostream>
#include <stdexcept>
#include "cnn/multichannel_conv.hpp"

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { ++failures; \
    std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n"; } } while (0)
#define CHECK_THROWS(expr) do { bool t = false; try { expr; } \
    catch (const std::invalid_argument&) { t = true; } \
    if (!t) { ++failures; std::cerr << "FAIL line " << __LINE__ << ": no throw: " #expr "\n"; } } while (0)

static bool near(double a, double b) { return std::fabs(a - b) < 1e-12; }

static void fill_seq(Tensor& t, double start = 1.0) {
    for (size_t i = 0; i < t.data.size(); ++i) t.data[i] = start + double(i);
}
static void fill_const(Tensor& t, double v) { for (auto& d : t.data) d = v; }

// 1 image, 1 channel, 1 filter: 3x3 input 1..9, 2x2 ones -> [12 16 24 28]
static void test_single_channel_baseline() {
    Tensor x(1, 1, 3, 3); fill_seq(x);
    Tensor f(1, 1, 2, 2); fill_const(f, 1.0);
    Tensor y = cnn::conv2d_multichannel(x, f);
    CHECK((y.shape == std::vector<size_t>{1, 1, 2, 2}));
    CHECK(near(y(0,0,0,0), 12)); CHECK(near(y(0,0,0,1), 16));
    CHECK(near(y(0,0,1,0), 24)); CHECK(near(y(0,0,1,1), 28));
}

// 2 channels: channel 1 = 2 x channel 0, ones filter on both -> 3 x single-channel result
static void test_multi_channel_sum() {
    Tensor x(1, 2, 3, 3);
    for (size_t i = 0; i < 9; ++i) { x.data[i] = i + 1; x.data[9 + i] = 2.0 * (i + 1); }
    Tensor f(1, 2, 2, 2); fill_const(f, 1.0);
    Tensor y = cnn::conv2d_multichannel(x, f);
    CHECK(near(y(0,0,0,0), 36)); CHECK(near(y(0,0,0,1), 48));
    CHECK(near(y(0,0,1,0), 72)); CHECK(near(y(0,0,1,1), 84));
}

// Channels weighted differently: filter weights (1, -1) -> channel0 - channel1 = -channel0
static void test_channel_weights() {
    Tensor x(1, 2, 3, 3);
    for (size_t i = 0; i < 9; ++i) { x.data[i] = i + 1; x.data[9 + i] = 2.0 * (i + 1); }
    Tensor f(1, 2, 2, 2);
    for (size_t i = 0; i < 4; ++i) { f.data[i] = 1.0; f.data[4 + i] = -1.0; }
    Tensor y = cnn::conv2d_multichannel(x, f);
    CHECK(near(y(0,0,0,0), -12)); CHECK(near(y(0,0,1,1), -28));
}

// 3 filters -> 3 output channels, filter k = (k+1) * ones
static void test_multi_filter() {
    Tensor x(1, 1, 3, 3); fill_seq(x);
    Tensor f(3, 1, 2, 2);
    for (size_t k = 0; k < 3; ++k)
        for (size_t i = 0; i < 4; ++i) f.data[k * 4 + i] = double(k + 1);
    Tensor y = cnn::conv2d_multichannel(x, f);
    CHECK((y.shape == std::vector<size_t>{1, 3, 2, 2}));
    for (size_t k = 0; k < 3; ++k) {
        CHECK(near(y(0,k,0,0), 12.0 * (k + 1)));
        CHECK(near(y(0,k,1,1), 28.0 * (k + 1)));
    }
}

// Batch of 2: second image is 10 x the first; results must scale and not mix.
static void test_batch() {
    Tensor x(2, 1, 3, 3);
    for (size_t i = 0; i < 9; ++i) { x.data[i] = i + 1; x.data[9 + i] = 10.0 * (i + 1); }
    Tensor f(1, 1, 2, 2); fill_const(f, 1.0);
    Tensor y = cnn::conv2d_multichannel(x, f);
    CHECK((y.shape == std::vector<size_t>{2, 1, 2, 2}));
    CHECK(near(y(0,0,0,0), 12)); CHECK(near(y(1,0,0,0), 120)); CHECK(near(y(1,0,1,1), 280));
}

// 1x1 kernel with weight 1 returns the input unchanged; full-size kernel gives 1x1 output.
static void test_edge_shapes() {
    Tensor x(1, 1, 4, 5); fill_seq(x);
    Tensor id(1, 1, 1, 1); id.data[0] = 1.0;
    Tensor y = cnn::conv2d_multichannel(x, id);
    CHECK((y.shape == std::vector<size_t>{1, 1, 4, 5}));
    for (size_t i = 0; i < x.data.size(); ++i) CHECK(near(y.data[i], x.data[i]));

    Tensor full(1, 1, 4, 5); fill_const(full, 1.0);
    Tensor z = cnn::conv2d_multichannel(x, full);   // sum of 1..20 = 210
    CHECK((z.shape == std::vector<size_t>{1, 1, 1, 1})); CHECK(near(z.data[0], 210));
}

static void test_invalid_inputs() {
    Tensor x(1, 2, 3, 3); Tensor f_bad_c(1, 3, 2, 2); Tensor f_big(1, 2, 4, 4);
    CHECK_THROWS(cnn::conv2d_multichannel(x, f_bad_c));   // channel mismatch
    CHECK_THROWS(cnn::conv2d_multichannel(x, f_big));     // kernel > input
    Tensor bad(1, 2, 3, 3); bad.shape.pop_back();         // not 4D
    Tensor ok(1, 2, 2, 2);
    CHECK_THROWS(cnn::conv2d_multichannel(bad, ok));
    Tensor corrupt(1, 2, 3, 3); corrupt.data.pop_back();  // data/shape mismatch
    CHECK_THROWS(cnn::conv2d_multichannel(corrupt, ok));
    CHECK_THROWS(cnn::conv_output_size(3, 0));
    CHECK_THROWS(cnn::conv_output_size(3, 4));
}

int main() {
    test_single_channel_baseline();
    test_multi_channel_sum();
    test_channel_weights();
    test_multi_filter();
    test_batch();
    test_edge_shapes();
    test_invalid_inputs();
    if (failures == 0) { std::cout << "All multichannel conv tests passed.\n"; return 0; }
    std::cerr << failures << " check(s) failed.\n";
    return 1;
}
