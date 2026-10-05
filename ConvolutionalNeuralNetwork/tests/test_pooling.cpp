#include "cnn/pooling.hpp"
#include <iostream>

static int failures = 0;
#define CHECK(cond)                                                   \
    do {                                                              \
        if (!(cond)) {                                                \
            std::cerr << "FAILED: " #cond " (line " << __LINE__ << ")\n"; \
            ++failures;                                               \
        }                                                             \
    } while (0)

#define CHECK_THROWS(expr)                                            \
    do {                                                              \
        bool threw = false;                                           \
        try { expr; } catch (const std::invalid_argument&) { threw = true; } \
        CHECK(threw);                                                 \
    } while (0)

void test_output_dimension_formula() {
    CHECK(cnn::pooled_dim(4, 2, 2) == 2);
    CHECK(cnn::pooled_dim(5, 2, 2) == 2);   // floor behaviour
    CHECK(cnn::pooled_dim(5, 3, 1) == 3);
}

void test_invalid_inputs() {
    CHECK_THROWS(cnn::pooled_dim(3, 5, 1));  // window > input
    CHECK_THROWS(cnn::pooled_dim(4, 0, 1));
    CHECK_THROWS(cnn::pooled_dim(4, 2, 0));
    cnn::PoolingParams bad;
    bad.stride_h = 0;
    CHECK_THROWS(bad.validate());
}

// ---- Stubs: fill in once Tensor and src/pooling.cpp exist (Week 2) ----
void test_max_pooling_forward()     { /* TODO */ }
void test_average_pooling_forward() { /* TODO */ }
void test_pooling_edge_cases()      { /* TODO: 1x1 input, window == input */ }

int main() {
    test_output_dimension_formula();
    test_invalid_inputs();
    test_max_pooling_forward();
    test_average_pooling_forward();
    test_pooling_edge_cases();
    if (failures == 0) std::cout << "All pooling tests passed\n";
    return failures == 0 ? 0 : 1;
}