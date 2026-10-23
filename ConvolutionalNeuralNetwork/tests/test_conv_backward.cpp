#include "conv_backward.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::cerr << "FAILED: " #cond " (line " << __LINE__ << ")\n"; ++failures; } } while (0)
#define CHECK_THROWS(expr) do { bool t = false; \
    try { expr; } catch (const std::invalid_argument&) { t = true; } CHECK(t); } while (0)

static bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) < tol; }

// Reference forward pass (cross-correlation, zero padding, stride).
// Replace with the team's Conv2D forward once it is on main.
static cnn::Tensor forward_ref(const cnn::Tensor& x, const cnn::Tensor& k,
                               std::size_t stride, std::size_t pad) {
    const std::size_t N = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
    const std::size_t K = k.shape[0], Kh = k.shape[2], Kw = k.shape[3];
    const std::size_t Ho = cnn::conv_out_size(H, Kh, stride, pad);
    const std::size_t Wo = cnn::conv_out_size(W, Kw, stride, pad);
    cnn::Tensor y(N, K, Ho, Wo);
    for (std::size_t n = 0; n < N; ++n)
      for (std::size_t f = 0; f < K; ++f)
        for (std::size_t i = 0; i < Ho; ++i)
          for (std::size_t j = 0; j < Wo; ++j) {
              double sum = 0.0;
              for (std::size_t c = 0; c < C; ++c)
                for (std::size_t r = 0; r < Kh; ++r)
                  for (std::size_t s = 0; s < Kw; ++s) {
                      long h = (long)(i * stride + r) - (long)pad;
                      long w = (long)(j * stride + s) - (long)pad;
                      if (h < 0 || h >= (long)H || w < 0 || w >= (long)W) continue;
                      sum += x(n, c, (size_t)h, (size_t)w) * k(f, c, r, s);
                  }
              y(n, f, i, j) = sum;
          }
    return y;
}

static void fill(cnn::Tensor& t, double seed) {
    for (std::size_t i = 0; i < t.data.size(); ++i) t.data[i] = std::sin(seed + 0.37 * i);
}

// Loss L = sum(Y * G), so dL/dY = G.
static double loss(const cnn::Tensor& y, const cnn::Tensor& g) {
    double s = 0.0;
    for (std::size_t i = 0; i < y.data.size(); ++i) s += y.data[i] * g.data[i];
    return s;
}

// Compare analytic gradients with central finite differences.
static void gradient_check(std::size_t N, std::size_t C, std::size_t H, std::size_t W,
                           std::size_t K, std::size_t Kh, std::size_t Kw,
                           std::size_t stride, std::size_t pad) {
    cnn::Tensor x(N, C, H, W), k(K, C, Kh, Kw);
    fill(x, 1.0); fill(k, 2.0);
    cnn::Tensor g(N, K, cnn::conv_out_size(H, Kh, stride, pad),
                        cnn::conv_out_size(W, Kw, stride, pad));
    fill(g, 3.0);

    cnn::ConvGradients grads = cnn::conv2d_backward(x, k, g, stride, pad);
    const double eps = 1e-6;

    for (std::size_t i = 0; i < x.data.size(); ++i) {
        const double orig = x.data[i];
        x.data[i] = orig + eps; const double lp = loss(forward_ref(x, k, stride, pad), g);
        x.data[i] = orig - eps; const double lm = loss(forward_ref(x, k, stride, pad), g);
        x.data[i] = orig;
        CHECK(near(grads.input_grad.data[i], (lp - lm) / (2 * eps), 1e-5));
    }
    for (std::size_t i = 0; i < k.data.size(); ++i) {
        const double orig = k.data[i];
        k.data[i] = orig + eps; const double lp = loss(forward_ref(x, k, stride, pad), g);
        k.data[i] = orig - eps; const double lm = loss(forward_ref(x, k, stride, pad), g);
        k.data[i] = orig;
        CHECK(near(grads.kernel_grad.data[i], (lp - lm) / (2 * eps), 1e-5));
    }
}

// Hand-checkable: 3x3 input (1..9), 2x2 kernel of ones, dY all ones.
void test_hand_computed() {
    cnn::Tensor x(1, 1, 3, 3), k(1, 1, 2, 2), g(1, 1, 2, 2);
    double v = 1;
    for (size_t h = 0; h < 3; ++h) for (size_t w = 0; w < 3; ++w) x(0,0,h,w) = v++;
    for (size_t r = 0; r < 2; ++r) for (size_t s = 0; s < 2; ++s) { k(0,0,r,s) = 1; g(0,0,r,s) = 1; }

    cnn::ConvGradients gr = cnn::conv2d_backward(x, k, g);
    // dK(r,s) = sum of the input window shifted by (r,s)
    CHECK(near(gr.kernel_grad(0,0,0,0), 12)); CHECK(near(gr.kernel_grad(0,0,0,1), 16));
    CHECK(near(gr.kernel_grad(0,0,1,0), 24)); CHECK(near(gr.kernel_grad(0,0,1,1), 28));
    // dX = how many windows cover each pixel
    const double expect[3][3] = {{1,2,1},{2,4,2},{1,2,1}};
    for (size_t h = 0; h < 3; ++h)
        for (size_t w = 0; w < 3; ++w) CHECK(near(gr.input_grad(0,0,h,w), expect[h][w]));
}

void test_gradient_checks() {
    gradient_check(1, 1, 5, 5, 1, 3, 3, 1, 0);   // basic
    gradient_check(2, 3, 6, 6, 2, 3, 3, 1, 1);   // batch, channels, filters, padding
    gradient_check(1, 2, 7, 7, 3, 3, 3, 2, 1);   // stride 2 + padding
    gradient_check(2, 1, 5, 5, 1, 2, 2, 2, 0);   // stride 2, input not fully covered
}

void test_invalid_inputs() {
    cnn::Tensor x(1, 2, 4, 4), k(1, 2, 3, 3), g(1, 1, 2, 2);
    CHECK_THROWS(cnn::conv2d_backward(x, k, g, 0, 0));          // stride 0
    cnn::Tensor bad_k(1, 3, 3, 3);
    CHECK_THROWS(cnn::conv2d_backward(x, bad_k, g));            // channel mismatch
    cnn::Tensor bad_g(1, 1, 3, 3);
    CHECK_THROWS(cnn::conv2d_backward(x, k, bad_g));            // wrong dY shape
    cnn::Tensor big_k(1, 2, 7, 7);
    CHECK_THROWS(cnn::conv2d_backward(x, big_k, g));            // kernel too large
}

int main() {
    test_hand_computed();
    test_gradient_checks();
    test_invalid_inputs();
    if (failures == 0) std::cout << "All conv backward tests passed\n";
    return failures == 0 ? 0 : 1;
}