
#include "pooling_backward.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace std;

int failures = 0;

#define CHECK(condition)                                      \
    do                                                        \
    {                                                         \
        if (!(condition))                                     \
        {                                                     \
            cerr << "FAILED: " #condition                    \
                 << " at line " << __LINE__ << '\n';          \
            failures++;                                       \
        }                                                     \
    } while (false)

#define CHECK_THROWS(statement)                               \
    do                                                        \
    {                                                         \
        bool threw = false;                                  \
        try                                                   \
        {                                                     \
            statement;                                        \
        }                                                     \
        catch (const invalid_argument&)                       \
        {                                                     \
            threw = true;                                     \
        }                                                     \
        CHECK(threw);                                         \
    } while (false)

int main()
{
    // Prepare a 4x4 input tensor.
    Tensor input(1, 1, 4, 4);

    double values[16] =
    {
         1,  2,  5,  4,
         3,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    for (size_t i = 0; i < 16; i++)
    {
        input.data[i] = values[i];
    }

    // Test 1: Max pooling backward with stride 2.
    MaxPoolingResult max_result = maxPool2D(input);

    Tensor max_upstream(1, 1, 2, 2);
    max_upstream(0, 0, 0, 0) = 1.0;
    max_upstream(0, 0, 0, 1) = 2.0;
    max_upstream(0, 0, 1, 0) = 3.0;
    max_upstream(0, 0, 1, 1) = 4.0;

    Tensor max_gradient = maxPool2DBackward(
        input, max_result, max_upstream
    );

    CHECK(max_gradient.shape == input.shape);
    CHECK(max_gradient(0, 0, 1, 1) == 1.0);
    CHECK(max_gradient(0, 0, 1, 3) == 2.0);
    CHECK(max_gradient(0, 0, 3, 1) == 3.0);
    CHECK(max_gradient(0, 0, 3, 3) == 4.0);
    CHECK(max_gradient(0, 0, 0, 0) == 0.0);

    // Test 2: Average pooling backward with stride 2.
    Tensor average_upstream(1, 1, 2, 2);
    average_upstream.fill(4.0);

    PoolingParams normal_params;

    Tensor average_gradient = averagePool2DBackward(
        input, normal_params, average_upstream
    );

    // Each 2x2 window distributes 4.0 / 4 = 1.0.
    CHECK(average_gradient(0, 0, 0, 0) == 1.0);
    CHECK(average_gradient(0, 0, 0, 1) == 1.0);
    CHECK(average_gradient(0, 0, 1, 0) == 1.0);
    CHECK(average_gradient(0, 0, 1, 1) == 1.0);
    CHECK(average_gradient(0, 0, 2, 2) == 1.0);

    // Test 3: Overlapping average pooling accumulates gradients.
    PoolingParams overlap_params;
    overlap_params.window_h = 2;
    overlap_params.window_w = 2;
    overlap_params.stride_h = 1;
    overlap_params.stride_w = 1;

    Tensor overlap_upstream(1, 1, 3, 3);
    overlap_upstream.fill(1.0);

    Tensor overlap_gradient = averagePool2DBackward(
        input, overlap_params, overlap_upstream
    );

    // Center input element belongs to four windows.
    CHECK(abs(overlap_gradient(0, 0, 1, 1) - 1.0) < 1e-9);

    // A corner belongs to only one window.
    CHECK(abs(overlap_gradient(0, 0, 0, 0) - 0.25) < 1e-9);

    // An edge element belongs to two windows.
    CHECK(abs(overlap_gradient(0, 0, 0, 1) - 0.5) < 1e-9);

    // Test 4: Reject a mismatched max-pooling gradient shape.
    Tensor wrong_gradient(1, 1, 1, 1);

    CHECK_THROWS(
        maxPool2DBackward(input, max_result, wrong_gradient)
    );

    // Test 5: Reject a zero average-pooling stride.
    PoolingParams invalid_params;
    invalid_params.stride_w = 0;

    CHECK_THROWS(
        averagePool2DBackward(input, invalid_params, average_upstream)
    );

    if (failures == 0)
    {
        cout << "All pooling backward tests passed!" << endl;
        return 0;
    }

    cout << failures << " test(s) failed." << endl;
    return 1;
}
