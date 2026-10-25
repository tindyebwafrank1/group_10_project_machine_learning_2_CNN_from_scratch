
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "preprocess.hpp"

using namespace std;
using namespace cnn;

using Vec = vector<double>;

static int failures = 0;

static void check_impl(
    bool ok,
    const char* expr,
    const char* file,
    int line
) {
    if (!ok) {
        cerr << "FAIL " << file << ":" << line
             << "  " << expr << "\n";
        ++failures;
    }
}

#define CHECK(cond) check_impl((cond), #cond, __FILE__, __LINE__)

template <typename F>
static bool throws(F&& f) {
    try {
        f();
    } catch (const invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }

    return false;
}

static bool approx(
    double a,
    double b,
    double eps = 1e-6
) {
    return fabs(a - b) <= eps;
}

static double at(
    const Tensor& t,
    size_t n,
    size_t c,
    size_t h,
    size_t w
) {
    return t.data[
        ((n * t.shape[1] + c) * t.shape[2] + h)
        * t.shape[3] + w
    ];
}

static bool shapeIs(
    const Tensor& t,
    size_t n,
    size_t c,
    size_t h,
    size_t w
) {
    return t.shape == vector<size_t>({n, c, h, w});
}

static Tensor make(
    size_t n,
    size_t c,
    size_t h,
    size_t w,
    const Vec& values
) {
    Tensor t(n, c, h, w);
    t.data = values;
    return t;
}

// A 2x2 RGB image:
// red, green
// blue, white
static const vector<unsigned char> kPx = {
    255, 0, 0,
    0, 255, 0,
    0, 0, 255,
    255, 255, 255
};

static Tensor rgb2x2() {
    return fromInterleaved(
        kPx.data(),
        kPx.size(),
        2,
        2,
        3
    );
}

// Test HWC to NCHW conversion.
static void test_from_interleaved() {
    const Tensor t = rgb2x2();

    CHECK(shapeIs(t, 1, 3, 2, 2));
    CHECK(at(t, 0, 0, 0, 0) == 255.0);
    CHECK(at(t, 0, 1, 0, 1) == 255.0);
    CHECK(at(t, 0, 2, 1, 0) == 255.0);
    CHECK(at(t, 0, 0, 1, 1) == 255.0);
    CHECK(at(t, 0, 1, 1, 0) == 0.0);

    const double d[3] = {1, 2, 3};

    CHECK(
        at(fromInterleaved(d, 3, 1, 1, 3),
           0, 2, 0, 0) == 3.0
    );

    CHECK(throws([&] {
        (void)fromInterleaved<unsigned char>(
            nullptr, 0, 2, 2, 3
        );
    }));

    CHECK(throws([&] {
        (void)fromInterleaved(
            kPx.data(), 11, 2, 2, 3
        );
    }));

    CHECK(throws([&] {
        (void)fromInterleaved(
            kPx.data(), 12, 0, 2, 3
        );
    }));

    CHECK(throws([&] {
        (void)fromInterleaved(
            kPx.data(), 12, 2, 2, 2
        );
    }));
}

// Test conversion back to interleaved HWC.
static void test_to_interleaved_roundtrip() {
    const Vec back = toInterleaved(rgb2x2(), 0);

    CHECK(back.size() == kPx.size());

    for (size_t i = 0; i < back.size(); ++i) {
        CHECK(back[i] == static_cast<double>(kPx[i]));
    }

    CHECK(throws([&] {
        (void)toInterleaved(rgb2x2(), 1);
    }));
}

// Test RGB/BGR conversion and arbitrary channel permutations.
static void test_channel_order() {
    const Tensor original = rgb2x2();
    const Tensor bgr = reverseChannels(original);

    CHECK(at(bgr, 0, 2, 0, 0) == 255.0);
    CHECK(at(bgr, 0, 0, 0, 0) == 0.0);
    CHECK(at(bgr, 0, 0, 1, 0) == 255.0);

    const Tensor again = reverseChannels(bgr);

    CHECK(again.data == original.data);

    CHECK(throws([&] {
        (void)permuteChannels(original, {0, 0, 1});
    }));

    CHECK(throws([&] {
        (void)permuteChannels(original, {0, 1});
    }));

    CHECK(throws([&] {
        (void)permuteChannels(original, {0, 1, 3});
    }));

    const Tensor p = permuteChannels(original, {1, 2, 0});

    CHECK(at(p, 0, 0, 0, 1) == 255.0);
}

// Test grayscale conversion and channel replication.
static void test_grayscale() {
    const Tensor g = rgbToGrayscale(rgb2x2());

    CHECK(shapeIs(g, 1, 1, 2, 2));
    CHECK(approx(at(g, 0, 0, 0, 0), 0.299 * 255));
    CHECK(approx(at(g, 0, 0, 0, 1), 0.587 * 255));
    CHECK(approx(at(g, 0, 0, 1, 0), 0.114 * 255));
    CHECK(approx(at(g, 0, 0, 1, 1), 255.0));

    const Tensor rgb = grayscaleToRgb(
        make(1, 1, 1, 2, {10, 20})
    );

    CHECK(shapeIs(rgb, 1, 3, 1, 2));

    for (size_t c = 0; c < 3; ++c) {
        CHECK(at(rgb, 0, c, 0, 0) == 10.0);
        CHECK(at(rgb, 0, c, 0, 1) == 20.0);
    }

    CHECK(shapeIs(toChannels(rgb2x2(), 3), 1, 3, 2, 2));
    CHECK(shapeIs(toChannels(rgb2x2(), 1), 1, 1, 2, 2));

    CHECK(throws([&] {
        (void)toChannels(rgb2x2(), 2);
    }));

    CHECK(throws([&] {
        (void)toChannels(make(1, 2, 1, 1, {1, 2}), 1);
    }));

    CHECK(throws([&] {
        (void)rgbToGrayscale(make(1, 1, 1, 1, {1}));
    }));

    CHECK(throws([&] {
        (void)grayscaleToRgb(rgb2x2());
    }));
}

// Regression test: grayscale must use RGB luminance weights,
// even when reverse_channels is enabled.
static void test_reverse_and_grayscale() {
    PreprocessConfig cfg;

    cfg.reverse_channels = true;
    cfg.target_channels = 1;
    cfg.normalization = Normalization::None;

    const Tensor result = preprocess(rgb2x2(), cfg);

    CHECK(shapeIs(result, 1, 1, 2, 2));

    CHECK(approx(
        at(result, 0, 0, 0, 0),
        0.299 * 255.0
    ));

    CHECK(approx(
        at(result, 0, 0, 0, 1),
        0.587 * 255.0
    ));

    CHECK(approx(
        at(result, 0, 0, 1, 0),
        0.114 * 255.0
    ));
}

// Test nearest-neighbor and bilinear interpolation.
static void test_resize() {
    Vec idx(16);

    for (size_t i = 0; i < 16; ++i) {
        idx[i] = static_cast<double>(i);
    }

    const Tensor dn = resize(
        make(1, 1, 4, 4, idx),
        2,
        2,
        Interpolation::Nearest
    );

    CHECK(shapeIs(dn, 1, 1, 2, 2));
    CHECK(at(dn, 0, 0, 0, 0) == 5.0);
    CHECK(at(dn, 0, 0, 0, 1) == 7.0);
    CHECK(at(dn, 0, 0, 1, 0) == 13.0);
    CHECK(at(dn, 0, 0, 1, 1) == 15.0);

    const Tensor up = resize(
        make(1, 1, 1, 2, {0, 100}),
        1,
        4
    );

    CHECK(approx(at(up, 0, 0, 0, 0), 0.0));
    CHECK(approx(at(up, 0, 0, 0, 1), 25.0));
    CHECK(approx(at(up, 0, 0, 0, 2), 75.0));
    CHECK(approx(at(up, 0, 0, 0, 3), 100.0));

    const Tensor average = resize(
        make(1, 1, 2, 2, {0, 100, 100, 200}),
        1,
        1
    );

    CHECK(approx(at(average, 0, 0, 0, 0), 100.0));

    Tensor constant(1, 2, 3, 3);
    constant.data.assign(18, 42.0);

    for (Interpolation mode : {
        Interpolation::Nearest,
        Interpolation::Bilinear
    }) {
        const Tensor r = resize(constant, 5, 7, mode);

        CHECK(shapeIs(r, 1, 2, 5, 7));

        for (double v : r.data) {
            CHECK(approx(v, 42.0));
        }
    }

    CHECK(throws([&] {
        (void)resize(constant, 0, 4);
    }));

    CHECK(throws([&] {
        (void)resize(constant, 4, 0);
    }));
}

// Test scaling into the 0..1 range.
static void test_scale() {
    const Tensor s = scaleToUnit(
        make(1, 1, 1, 3, {0, 127.5, 255})
    );

    CHECK(approx(s.data[0], 0.0));
    CHECK(approx(s.data[1], 0.5));
    CHECK(approx(s.data[2], 1.0));

    CHECK(throws([&] {
        (void)scaleToUnit(s, 0.0);
    }));

    CHECK(throws([&] {
        (void)scaleToUnit(s, -1.0);
    }));

    CHECK(throws([&] {
        (void)scaleToUnit(make(1, 1, 1, 1, {nan("")}));
    }));
}

// Test per-channel standardization.
static void test_standardize() {
    const Tensor t = make(
        1, 2, 1, 2,
        {0, 2, 10, 30}
    );

    const Tensor s = standardize(
        t,
        {1, 20},
        {1, 10}
    );

    CHECK(approx(s.data[0], -1.0));
    CHECK(approx(s.data[1], 1.0));
    CHECK(approx(s.data[2], -1.0));
    CHECK(approx(s.data[3], 1.0));

    CHECK(throws([&] {
        (void)standardize(t, {1}, {1, 10});
    }));

    CHECK(throws([&] {
        (void)standardize(t, {1, 20}, {1, 0});
    }));

    CHECK(throws([&] {
        (void)standardize(t, {1, 20}, {1, -2});
    }));
}

// Test min-max normalization.
static void test_minmax() {
    const Tensor t = make(
        2, 1, 1, 3,
        {10, 20, 30, 5, 5, 5}
    );

    const Tensor a = minMaxNormalize(t);

    CHECK(approx(a.data[0], 0.0));
    CHECK(approx(a.data[1], 0.5));
    CHECK(approx(a.data[2], 1.0));

    CHECK(approx(a.data[3], 0.0));
    CHECK(approx(a.data[4], 0.0));
    CHECK(approx(a.data[5], 0.0));

    const Tensor b = minMaxNormalize(t, -1.0, 1.0);

    CHECK(approx(b.data[0], -1.0));
    CHECK(approx(b.data[1], 0.0));
    CHECK(approx(b.data[2], 1.0));
    CHECK(approx(b.data[3], -1.0));

    CHECK(throws([&] {
        (void)minMaxNormalize(t, 1.0, 1.0);
    }));

    CHECK(throws([&] {
        (void)minMaxNormalize(t, 2.0, 1.0);
    }));
}

// Test population statistics for each channel.
static void test_stats() {
    const ChannelStats s = computeChannelStats(
        make(1, 2, 1, 2, {0, 2, 4, 4})
    );

    CHECK(approx(s.mean[0], 1.0));
    CHECK(approx(s.stddev[0], 1.0));

    CHECK(approx(s.mean[1], 4.0));
    CHECK(approx(s.stddev[1], 1.0));
}

// Test batch stacking.
static void test_stack_batch() {
    const Tensor a = make(1, 1, 1, 2, {1, 2});
    const Tensor b = make(1, 1, 1, 2, {3, 4});

    const Tensor s = stackBatch({a, b});

    CHECK(shapeIs(s, 2, 1, 1, 2));
    CHECK(s.data == Vec({1, 2, 3, 4}));

    CHECK(throws([&] {
        (void)stackBatch({
            a,
            make(1, 1, 1, 3, {1, 2, 3})
        });
    }));

    CHECK(throws([&] {
        (void)stackBatch({});
    }));
}

// Test the complete preprocessing pipeline.
static void test_pipeline() {
    const Tensor original = rgb2x2();

    // Default configuration: scale to [0,1].
    const Tensor d = preprocess(original, PreprocessConfig{});

    CHECK(shapeIs(d, 1, 3, 2, 2));
    CHECK(approx(at(d, 0, 0, 0, 0), 1.0));

    for (double v : d.data) {
        CHECK(v >= 0.0 && v <= 1.0);
    }

    // Grayscale, nearest-neighbor resize and scaling.
    PreprocessConfig g;

    g.target_channels = 1;
    g.target_height = 4;
    g.target_width = 4;
    g.interpolation = Interpolation::Nearest;

    const Tensor gt = preprocess(original, g);

    CHECK(shapeIs(gt, 1, 1, 4, 4));
    CHECK(approx(at(gt, 0, 0, 0, 0), 0.299));
    CHECK(approx(at(gt, 0, 0, 3, 3), 1.0));

    // Reverse RGB to BGR and standardize.
    PreprocessConfig s;

    s.reverse_channels = true;
    s.normalization = Normalization::Standardize;
    s.mean = {0.5, 0.5, 0.5};
    s.stddev = {0.5, 0.5, 0.5};

    const Tensor st = preprocess(original, s);

    CHECK(approx(at(st, 0, 2, 0, 0), 1.0));
    CHECK(approx(at(st, 0, 0, 0, 0), -1.0));

    // Min-max normalization.
    PreprocessConfig m;
    m.normalization = Normalization::MinMax;

    CHECK(approx(
        at(preprocess(original, m), 0, 0, 0, 0),
        1.0
    ));

    // No normalization preserves the original values.
    PreprocessConfig n;
    n.normalization = Normalization::None;

    CHECK(approx(
        at(preprocess(original, n), 0, 0, 0, 0),
        255.0
    ));

    // Input must remain unchanged.
    CHECK(original.data == preprocess(original, n).data);

    // Resize a non-square batch.
    Tensor big(2, 3, 2, 5);
    big.data.assign(60, 10.0);

    PreprocessConfig r;
    r.target_height = 4;
    r.target_width = 3;

    CHECK(shapeIs(preprocess(big, r), 2, 3, 4, 3));
}

// Test invalid pipeline configurations and tensors.
static void test_pipeline_invalid() {
    PreprocessConfig a;
    a.target_height = 4;

    CHECK(throws([&] {
        (void)preprocess(rgb2x2(), a);
    }));

    PreprocessConfig b;
    b.target_channels = 2;

    CHECK(throws([&] {
        (void)preprocess(rgb2x2(), b);
    }));

    PreprocessConfig c;
    c.normalization = Normalization::Standardize;
    c.mean = {0.5, 0.5};
    c.stddev = {0.5, 0.5};

    CHECK(throws([&] {
        (void)preprocess(rgb2x2(), c);
    }));

    PreprocessConfig d;
    d.max_value = 0.0;

    CHECK(throws([&] {
        (void)preprocess(rgb2x2(), d);
    }));

    CHECK(throws([&] {
        (void)preprocess(
            make(1, 1, 1, 1, {nan("")}),
            PreprocessConfig{}
        );
    }));

    Tensor broken(1, 1, 2, 2);
    broken.data.pop_back();

    CHECK(throws([&] {
        (void)preprocess(broken, PreprocessConfig{});
    }));

    Tensor flat3d(1, 1, 1, 1);
    flat3d.shape.pop_back();

    CHECK(throws([&] {
        (void)preprocess(flat3d, PreprocessConfig{});
    }));
}

int main() {
    test_from_interleaved();
    test_to_interleaved_roundtrip();
    test_channel_order();
    test_grayscale();
    test_reverse_and_grayscale();
    test_resize();
    test_scale();
    test_standardize();
    test_minmax();
    test_stats();
    test_stack_batch();
    test_pipeline();
    test_pipeline_invalid();

    if (failures == 0) {
        cout << "test_preprocess: all passed" << endl;
    }

    return failures == 0 ? 0 : 1;
}
