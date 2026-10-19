#include "convolution.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::cerr << "FAILED: " #cond " (line " << __LINE__ << ")\n"; ++failures; } } while (0)

static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

static cnn::Tensor make_3x3() {            // values 1..9
    cnn::Tensor x(1, 1, 3, 3);
    double v = 1;
    for (size_t h = 0; h < 3; ++h)
        for (size_t w = 0; w < 3; ++w) x(0, 0, h, w) = v++;
    return x;
}

void test_output_dim() {
    CHECK(cnn::conv_output_dim(5, 3) == 3);
    CHECK(cnn::conv_output_dim(3, 3) == 1);
}

void test_all_ones_kernel() {              // window sums
    cnn::Tensor x = make_3x3(), k(1, 1, 2, 2);
    for (size_t r = 0; r < 2; ++r)
        for (size_t s = 0; s < 2; ++s) k(0, 0, r, s) = 1.0;
    cnn::Tensor y = cnn::conv2d_single(x, k);
    CHECK(y.shape[2] == 2 && y.shape[3] == 2);
    CHECK(near(y(0,0,0,0), 12)); CHECK(near(y(0,0,0,1), 16));
    CHECK(near(y(0,0,1,0), 24)); CHECK(near(y(0,0,1,1), 28));
}

void test_diagonal_kernel() {              // [1 0; 0 -1] gives -4 everywhere
    cnn::Tensor x = make_3x3(), k(1, 1, 2, 2);
    k(0,0,0,0) = 1; k(0,0,0,1) = 0; k(0,0,1,0) = 0; k(0,0,1,1) = -1;
    cnn::Tensor y = cnn::conv2d_single(x, k);
    for (size_t h = 0; h < 2; ++h)
        for (size_t w = 0; w < 2; ++w) CHECK(near(y(0,0,h,w), -4));
}

void test_invalid_inputs() {
    cnn::Tensor x = make_3x3(), big(1, 1, 5, 5);
    bool threw = false;
    try { cnn::conv2d_single(x, big); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

int main() {
    test_output_dim();
    test_all_ones_kernel();
    test_diagonal_kernel();
    test_invalid_inputs();
    if (failures == 0) std::cout << "All convolution tests passed\n";
    return failures == 0 ? 0 : 1;
}