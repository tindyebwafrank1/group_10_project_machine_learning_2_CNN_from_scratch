// Plain-assert tests, no external dependencies.
#include <cmath>
#include <iostream>
#include "cnn_library/metrics.hpp"

using namespace CNN;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { ++failures; \
    std::cerr << "FAIL line " << __LINE__ << ": " #cond "\n"; } } while (0)
#define CHECK_NEAR(a, b) CHECK(std::fabs((a) - (b)) < 1e-9)
#define CHECK_THROWS(expr, ex) do { bool t = false; try { expr; } catch (const ex&) { t = true; } \
    CHECK(t); } while (0)

// 3-class example: actual = 0 0 0 1 1 1 2 2 2 2 ; predicted = 0 0 1 1 1 2 2 2 2 0
static ConfusionMatrix sample() {
    return build_confusion_matrix({0,0,0,1,1,1,2,2,2,2}, {0,0,1,1,1,2,2,2,2,0}, 3);
}

void test_counts() {
    ConfusionMatrix cm = sample();
    CHECK(cm.total() == 10);
    CHECK(cm.count(0,0) == 2); CHECK(cm.count(0,1) == 1);
    CHECK(cm.count(1,1) == 2); CHECK(cm.count(1,2) == 1);
    CHECK(cm.count(2,2) == 3); CHECK(cm.count(2,0) == 1);
    CHECK(cm.true_positives(2) == 3);
    CHECK(cm.false_positives(2) == 1);
    CHECK(cm.false_negatives(2) == 1);
    CHECK(cm.true_negatives(2) == 5);
    CHECK(cm.support(0) == 3);
}

void test_metrics() {
    ConfusionMatrix cm = sample();
    CHECK_NEAR(cm.accuracy(), 0.7);
    CHECK_NEAR(cm.precision(0), 2.0/3.0);
    CHECK_NEAR(cm.recall(0), 2.0/3.0);
    CHECK_NEAR(cm.precision(2), 0.75);
    CHECK_NEAR(cm.recall(2), 0.75);
    CHECK_NEAR(cm.f1(1), 2.0/3.0);
    CHECK_NEAR(cm.macro_precision(), (2.0/3.0 + 2.0/3.0 + 0.75) / 3.0);
    CHECK_NEAR(accuracy({0,1,1}, {0,1,0}), 2.0/3.0);
}

void test_perfect_and_worst() {
    ConfusionMatrix p = build_confusion_matrix({0,1,2}, {0,1,2}, 3);
    CHECK_NEAR(p.accuracy(), 1.0); CHECK_NEAR(p.macro_f1(), 1.0);
    ConfusionMatrix w = build_confusion_matrix({0,1,2}, {1,2,0}, 3);
    CHECK_NEAR(w.accuracy(), 0.0); CHECK_NEAR(w.macro_f1(), 0.0);
}

void test_zero_division() {
    ConfusionMatrix empty(3);
    CHECK_NEAR(empty.accuracy(), 0.0);
    CHECK_NEAR(empty.precision(0), 0.0);
    CHECK_NEAR(empty.recall(0), 0.0);
    CHECK_NEAR(empty.f1(0), 0.0);
    // class 2 never predicted -> precision 0, not NaN
    ConfusionMatrix cm = build_confusion_matrix({0,1,2}, {0,1,0}, 3);
    CHECK_NEAR(cm.precision(2), 0.0);
    CHECK_NEAR(accuracy({}, {}), 0.0);
}

void test_invalid_inputs() {
    CHECK_THROWS(ConfusionMatrix(0), std::invalid_argument);
    ConfusionMatrix cm(2);
    CHECK_THROWS(cm.add(2, 0), std::out_of_range);
    CHECK_THROWS(cm.add(0, 5), std::out_of_range);
    CHECK_THROWS(cm.precision(2), std::out_of_range);
    CHECK_THROWS(cm.add_batch({0,1}, {0}), std::invalid_argument);
    CHECK_THROWS(accuracy({0,1}, {0}), std::invalid_argument);
    // bad batch must not partially update
    CHECK_THROWS(cm.add_batch({0,1}, {0,9}), std::out_of_range);
    CHECK(cm.total() == 0);
}

void test_reset() {
    ConfusionMatrix cm = sample();
    cm.reset();
    CHECK(cm.total() == 0); CHECK(cm.count(0,0) == 0);
}

int main() {
    test_counts(); test_metrics(); test_perfect_and_worst();
    test_zero_division(); test_invalid_inputs(); test_reset();
    if (failures == 0) { std::cout << "All metrics tests passed.\n"; return 0; }
    std::cerr << failures << " check(s) failed.\n";
    return 1;
}
