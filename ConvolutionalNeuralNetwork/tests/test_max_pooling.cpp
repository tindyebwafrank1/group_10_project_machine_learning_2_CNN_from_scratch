
#include "max_pooling.hpp"

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
    // Test 1: Standard 2x2 max pooling with stride 2.
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

    MaxPoolingResult result = maxPool2D(input);

    CHECK(result.output.shape[0] == 1);
    CHECK(result.output.shape[1] == 1);
    CHECK(result.output.shape[2] == 2);
    CHECK(result.output.shape[3] == 2);

    CHECK(result.output(0, 0, 0, 0) == 6);
    CHECK(result.output(0, 0, 0, 1) == 8);
    CHECK(result.output(0, 0, 1, 0) == 14);
    CHECK(result.output(0, 0, 1, 1) == 16);

    // Maxima are at flat input indices 5, 7, 13 and 15.
    CHECK(result.max_indices.size() == 4);
    CHECK(result.max_indices[0] == 5);
    CHECK(result.max_indices[1] == 7);
    CHECK(result.max_indices[2] == 13);
    CHECK(result.max_indices[3] == 15);

    // Test 2: A custom window and stride.
    PoolingParams params;
    params.window_h = 2;
    params.window_w = 2;
    params.stride_h = 1;
    params.stride_w = 1;

    MaxPoolingResult overlapping = maxPool2D(input, params);

    CHECK(overlapping.output.shape[2] == 3);
    CHECK(overlapping.output.shape[3] == 3);
    CHECK(overlapping.output(0, 0, 0, 0) == 6);
    CHECK(overlapping.output(0, 0, 1, 1) == 11);

    // Test 3: Multiple channels remain independent.
    Tensor multi_channel(1, 2, 2, 2);

    multi_channel(0, 0, 0, 0) = 1;
    multi_channel(0, 0, 0, 1) = 2;
    multi_channel(0, 0, 1, 0) = 3;
    multi_channel(0, 0, 1, 1) = 4;

    multi_channel(0, 1, 0, 0) = 10;
    multi_channel(0, 1, 0, 1) = 20;
    multi_channel(0, 1, 1, 0) = 30;
    multi_channel(0, 1, 1, 1) = 40;

    MaxPoolingResult multi_result = maxPool2D(multi_channel);

    CHECK(multi_result.output(0, 0, 0, 0) == 4);
    CHECK(multi_result.output(0, 1, 0, 0) == 40);
    CHECK(multi_result.max_indices[0] == 3);
    CHECK(multi_result.max_indices[1] == 7);

    // Test 4: Invalid stride must be rejected.
    PoolingParams bad_stride;
    bad_stride.stride_h = 0;

    CHECK_THROWS(maxPool2D(input, bad_stride));

    // Test 5: A pooling window larger than the input is invalid.
    PoolingParams bad_window;
    bad_window.window_h = 5;

    CHECK_THROWS(maxPool2D(input, bad_window));

    // Test 6: Zero window size must be rejected.
    PoolingParams zero_window;
    zero_window.window_w = 0;

    CHECK_THROWS(maxPool2D(input, zero_window));

    if (failures == 0)
    {
        cout << "All max pooling tests passed!" << endl;
        return 0;
    }

    cout << failures << " test(s) failed." << endl;
    return 1;
}
