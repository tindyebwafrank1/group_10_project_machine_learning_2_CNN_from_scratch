#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "strides.hpp"

using namespace std;
using namespace cnn;
using Vec = vector<double>;
using Win = array<size_t, 4>;  // {oh, ow, padded_row, padded_col}

static int failures = 0;
static void check_impl(bool ok, const char* expr, const char* file, int line) {
    if (!ok) {
        cerr << "FAIL " << file << ":" << line << "  " << expr << "\n";
        ++failures;
    }
}
#define CHECK(cond) check_impl((cond), #cond, __FILE__, __LINE__)

template <typename F>
static bool throws(F&& f) {
    try { f(); } catch (const invalid_argument&) { return true; } catch (...) { return false; }
    return false;
}

static bool approx(double a, double b, double eps = 1e-9) { return fabs(a - b) <= eps; }

static bool shapeIs(const Tensor& t, size_t n, size_t c, size_t h, size_t w) {
    return t.shape == vector<size_t>({n, c, h, w});
}
static Tensor make(size_t n, size_t c, size_t h, size_t w, const Vec& v) {
    Tensor t(n, c, h, w);
    t.data = v;
    return t;
}
static Vec iota(size_t n) {
    Vec v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<double>(i);
    return v;
}


// 1. Output-size equation: hand-computed (H, W, P, S, K) combinations.
//    out = floor((in - K + 2P) / S) + 1, square kernel K, same P and S on both axes.

struct Combo { size_t H, W, P, S, K, outH, outW; };
static const Combo kCombos[] = {
    {5, 5, 0, 1, 3, 3, 3},      {5, 5, 0, 2, 3, 2, 2},      {5, 5, 1, 1, 3, 5, 5},
    {5, 5, 1, 2, 3, 3, 3},      {6, 6, 0, 2, 2, 3, 3},      {6, 6, 0, 3, 2, 2, 2},
    {7, 7, 0, 2, 3, 3, 3},      {7, 7, 1, 2, 3, 4, 4},      {7, 7, 2, 1, 5, 7, 7},
    {8, 8, 0, 2, 2, 4, 4},      {8, 6, 0, 2, 2, 4, 3},      {10, 10, 0, 3, 3, 3, 3},
    {10, 10, 1, 3, 3, 4, 4},    {28, 28, 0, 1, 5, 24, 24},  {28, 28, 2, 1, 5, 28, 28},
    {32, 32, 0, 2, 2, 16, 16},  {32, 32, 1, 2, 3, 16, 16},  {224, 224, 3, 2, 7, 112, 112},
    {3, 3, 0, 1, 3, 1, 1},      {3, 3, 0, 2, 3, 1, 1},      {4, 9, 1, 2, 3, 2, 5},
    {1, 1, 1, 1, 3, 1, 1},      {2, 2, 0, 5, 2, 1, 1},
};

static void test_formula_combinations() {
    const size_t count = sizeof(kCombos) / sizeof(kCombos[0]);
    CHECK(count >= 15);  // acceptance criterion: at least 15 distinct combinations
    for (const Combo& c : kCombos) {
        CHECK(outputSize(c.H, c.K, c.P, c.S) == c.outH);
        CHECK(outputSize(c.W, c.K, c.P, c.S) == c.outW);
        const WindowGrid g = makeWindowGrid(c.H, c.W, c.K, c.K, c.P, Stride(c.S));
        CHECK(g.out_h == c.outH && g.out_w == c.outW);
        const vector<size_t> shp = windowedOutputShape({2, 3, c.H, c.W}, c.K, c.K, c.P, Stride(c.S));
        CHECK(shp == vector<size_t>({2, 3, c.outH, c.outW}));
    }
}


// 2. Exhaustive check against a brute-force count of window positions, plus
//    the "last window fits, next one would not" property.

static void test_sweep_against_brute_force() {
    for (size_t in = 1; in <= 12; ++in) {
        for (size_t pad = 0; pad <= 3; ++pad) {
            for (size_t stride = 1; stride <= 6; ++stride) {
                for (size_t kernel = 1; kernel <= in + 2 * pad; ++kernel) {
                    const size_t padded = in + 2 * pad;
                    size_t brute = 0;
                    for (size_t p = 0; p * stride + kernel <= padded; ++p) ++brute;
                    const size_t formula = outputSize(in, kernel, pad, stride);
                    CHECK(formula == brute);
                    CHECK((formula - 1) * stride + kernel <= padded);  // last window fits
                    CHECK(formula * stride + kernel > padded);         // one more would not
                    CHECK(subsampleSize(in, stride) == outputSize(in, 1, 0, stride));
                }
            }
        }
    }
}

// 3. Invalid configurations

static void test_invalid_configurations() {
    CHECK(throws([&] { (void)outputSize(5, 3, 0, 0); }));                       // stride 0
    CHECK(throws([&] { (void)outputSize(5, 0, 0, 1); }));                       // kernel 0
    CHECK(throws([&] { (void)outputSize(0, 3, 0, 1); }));                       // empty input
    CHECK(throws([&] { (void)outputSize(3, 5, 0, 1); }));                       // kernel > input
    CHECK(throws([&] { (void)outputSize(5, 3, static_cast<size_t>(-1), 1); })); // huge padding

    string why;
    CHECK(!isValidWindow(5, 3, 0, 0, &why));
    CHECK(!why.empty());
    CHECK(!isValidWindow(3, 5, 0, 1));
    CHECK(isValidWindow(5, 3, 1, 2));
    CHECK(isValidWindow(1, 3, 1, 1));  // padding makes a 3-wide window fit a 1-wide input

    CHECK(throws([&] { (void)makeWindowGrid(5, 5, 3, 3, 0, Stride(0)); }));
    CHECK(throws([&] { (void)makeWindowGrid(5, 5, 3, 3, 0, Stride(2, 0)); }));
    CHECK(throws([&] { (void)windowedOutputShape({3, 8, 8}, 3, 3, 0, Stride(1)); }));      // 3-D
    CHECK(throws([&] { (void)windowedOutputShape({0, 3, 8, 8}, 3, 3, 0, Stride(1)); }));   // N = 0
}
//traversing the input tensor with a sliding window of size kernel_h x kernel_w, with padding pad and stride stride.
static void test_traversal() {
    const WindowGrid g = makeWindowGrid(5, 5, 3, 3, 0, Stride(2));
    CHECK(g.out_h == 2 && g.out_w == 2);
    vector<Win> got;
    forEachWindow(g, [&](size_t oh, size_t ow, size_t h0, size_t w0) { got.push_back(Win{oh, ow, h0, w0}); });
    const vector<Win> want = {Win{0, 0, 0, 0}, Win{0, 1, 0, 2}, Win{1, 0, 2, 0}, Win{1, 1, 2, 2}};
    CHECK(got == want);  // row-major order

    const WindowGrid p = makeWindowGrid(5, 5, 3, 3, 1, Stride(2));
    CHECK(p.out_h == 3 && p.out_w == 3);
    CHECK(p.inputRow(0) == -1 && p.inputRow(1) == 1 && p.inputRow(2) == 3);
    CHECK(p.inputCol(0) == -1 && p.inputCol(2) == 3);
    CHECK(p.paddedRow(2) == 4 && p.paddedCol(1) == 2);

    const WindowGrid u = makeWindowGrid(6, 8, 2, 2, 0, Stride(2, 4));  // different strides
    CHECK(u.out_h == 3 && u.out_w == 2);
    CHECK(u.paddedRow(2) == 4 && u.paddedCol(1) == 4);

    const WindowGrid s = makeWindowGrid(4, 4, 3, 3, 1, 0, Stride(1));  // different paddings
    CHECK(s.out_h == 4 && s.out_w == 2);
    CHECK(s.inputRow(0) == -1 && s.inputCol(0) == 0);

    size_t visited = 0;
    forEachWindow(p, [&](size_t, size_t, size_t, size_t) { ++visited; });
    CHECK(visited == 9);
}

static void test_padded_to_input() {
    const WindowGrid g = makeWindowGrid(5, 5, 3, 3, 1, Stride(2));
    size_t r = 99, c = 99;
    CHECK(!paddedToInput(g, 0, 0, r, c));          // corner of the padding
    CHECK(r == 99 && c == 99);                     // untouched on failure
    CHECK(paddedToInput(g, 1, 1, r, c) && r == 0 && c == 0);
    CHECK(paddedToInput(g, 5, 5, r, c) && r == 4 && c == 4);
    CHECK(!paddedToInput(g, 6, 3, r, c));          // bottom padding row
    CHECK(!paddedToInput(g, 3, 6, r, c));          // right padding column
    CHECK(!paddedToInput(g, 0, 3, r, c));          // top padding row
}

// 5. Integration: a convolution written with the grid equals one written with an
//    explicitly padded copy and hand-computed indices.

static void test_convolution_matches_explicit_padding() {
    const size_t H = 7, W = 6, KH = 3, KW = 3;
    for (size_t pad = 0; pad <= 2; ++pad) {
        for (size_t s = 1; s <= 3; ++s) {
            Vec x(H * W), k(KH * KW);
            for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<double>((i * 7) % 11) - 5.0;
            for (size_t i = 0; i < k.size(); ++i) k[i] = static_cast<double>((i * 3) % 5) - 2.0;

            // Reference: explicit zero-padded copy, direct index arithmetic.
            const size_t PH = H + 2 * pad, PW = W + 2 * pad;
            Vec padded(PH * PW, 0.0);
            for (size_t r = 0; r < H; ++r)
                for (size_t c = 0; c < W; ++c) padded[(r + pad) * PW + (c + pad)] = x[r * W + c];
            const size_t oh = (PH - KH) / s + 1, ow = (PW - KW) / s + 1;
            Vec ref(oh * ow, 0.0);
            for (size_t i = 0; i < oh; ++i)
                for (size_t j = 0; j < ow; ++j) {
                    double sum = 0.0;
                    for (size_t r = 0; r < KH; ++r)
                        for (size_t c = 0; c < KW; ++c) sum += padded[(i * s + r) * PW + (j * s + c)] * k[r * KW + c];
                    ref[i * ow + j] = sum;
                }

            // Same convolution through WindowGrid, with no padded copy.
            const WindowGrid g = makeWindowGrid(H, W, KH, KW, pad, Stride(s));
            CHECK(g.out_h == oh && g.out_w == ow);
            Vec got(g.out_h * g.out_w, 0.0);
            forEachWindow(g, [&](size_t i, size_t j, size_t h0, size_t w0) {
                double sum = 0.0;
                for (size_t r = 0; r < KH; ++r)
                    for (size_t c = 0; c < KW; ++c) {
                        size_t ir = 0, ic = 0;
                        if (paddedToInput(g, h0 + r, w0 + c, ir, ic)) sum += x[ir * W + ic] * k[r * KW + c];
                    }
                got[i * g.out_w + j] = sum;
            });
            for (size_t i = 0; i < got.size(); ++i) CHECK(approx(got[i], ref[i]));
        }
    }
}
// sub sampling: keep every stride-th element, starting at offset_h, offset_w. 
//The input is not modified.
static void test_subsample() {
    const Tensor t = make(1, 1, 4, 4, iota(16));

    const Tensor a = subsample(t, Stride(2));
    CHECK(shapeIs(a, 1, 1, 2, 2));
    CHECK(a.data == Vec({0, 2, 8, 10}));

    const Tensor b = subsample(t, Stride(1, 2));  // keep all rows, every 2nd column
    CHECK(shapeIs(b, 1, 1, 4, 2));
    CHECK(b.data == Vec({0, 2, 4, 6, 8, 10, 12, 14}));

    const Tensor c = subsample(t, Stride(2), 1, 1);  // start at (1,1)
    CHECK(shapeIs(c, 1, 1, 2, 2));
    CHECK(c.data == Vec({5, 7, 13, 15}));

    const Tensor d = subsample(make(1, 1, 5, 5, iota(25)), Stride(2));  // size not divisible
    CHECK(shapeIs(d, 1, 1, 3, 3));
    CHECK(d.data == Vec({0, 2, 4, 10, 12, 14, 20, 22, 24}));

    const Tensor e = subsample(make(1, 1, 4, 4, iota(16)), Stride(1));  // stride 1 = copy
    CHECK(e.data == t.data);

    Tensor m(2, 2, 4, 4);
    m.data = iota(64);
    const Tensor f = subsample(m, Stride(2));
    CHECK(shapeIs(f, 2, 2, 2, 2));
    CHECK(f.data[0] == 0.0 && f.data[1] == 2.0);
    CHECK(f.data[15] == 58.0);  // n=1,c=1,i=1,j=1 -> source (1,1,2,2) = 58

    CHECK(t.data == iota(16));  // input untouched

    CHECK(throws([&] { (void)subsample(t, Stride(0)); }));
    CHECK(throws([&] { (void)subsample(t, Stride(2), 4, 0); }));  // offset == H
    CHECK(throws([&] { (void)subsample(t, Stride(2), 0, 9); }));
    Tensor broken(1, 1, 2, 2);
    broken.data.pop_back();
    CHECK(throws([&] { (void)subsample(broken, Stride(1)); }));
    Tensor flat3d(1, 1, 1, 1);
    flat3d.shape.pop_back();
    CHECK(throws([&] { (void)subsample(flat3d, Stride(1)); }));
}

int main() {
    test_formula_combinations();
    test_sweep_against_brute_force();
    test_invalid_configurations();
    test_traversal();
    test_padded_to_input();
    test_convolution_matches_explicit_padding();
    test_subsample();
    if (failures == 0) cout << "test_stride: all passed\n";
    return failures == 0 ? 0 : 1;
}