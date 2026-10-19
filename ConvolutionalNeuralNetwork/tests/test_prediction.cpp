#include "prediction.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::cerr << "FAILED: " #cond " (line " << __LINE__ << ")\n"; ++failures; } } while (0)

#define CHECK_THROWS(expr) do { bool t = false; \
    try { expr; } catch (const std::invalid_argument&) { t = true; } CHECK(t); } while (0)

void test_argmax_basic() {
    CHECK(cnn::argmax({0.1, 0.7, 0.2}) == 1);
    CHECK(cnn::argmax({5.0}) == 0);                 // single element
    CHECK(cnn::argmax({-3.0, -1.0, -2.0}) == 1);    // all negative
}

void test_argmax_ties() {
    CHECK(cnn::argmax({0.4, 0.4, 0.2}) == 0);       // first index wins
}

void test_argmax_invalid() {
    CHECK_THROWS(cnn::argmax({}));
    CHECK_THROWS(cnn::argmax({0.5, std::nan(""), 0.1}));
}

void test_predict_batch() {
    cnn::Tensor s(2, 3, 1, 1);                      // 2 samples, 3 classes
    s.data = {0.2, 0.5, 0.3,                        // sample 0 -> class 1
              0.9, 0.05, 0.05};                     // sample 1 -> class 0
    auto labels = cnn::predict(s);
    CHECK(labels.size() == 2);
    CHECK(labels[0] == 1 && labels[1] == 0);
}

void test_predict_invalid() {
    cnn::Tensor empty(0, 3, 1, 1);
    CHECK_THROWS(cnn::predict(empty));
}

int main() {
    test_argmax_basic();
    test_argmax_ties();
    test_argmax_invalid();
    test_predict_batch();
    test_predict_invalid();
    if (failures == 0) std::cout << "All prediction tests passed\n";
    return failures == 0 ? 0 : 1;
}