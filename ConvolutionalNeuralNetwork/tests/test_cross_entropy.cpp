
#include "cross_entropy.hpp"

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
    // Test 1: Basic one-hot classification loss.
    vector<double> probabilities = {0.1, 0.7, 0.2};
    vector<double> targets = {0.0, 1.0, 0.0};

    double loss = crossEntropyLoss(probabilities, targets);

    CHECK(abs(loss - (-log(0.7))) < 1e-9);

    // Test 2: A correct prediction with high confidence.
    vector<double> good_probabilities = {0.01, 0.98, 0.01};

    double good_loss = crossEntropyLoss(
        good_probabilities, targets
    );

    CHECK(good_loss < loss);

    // Test 3: A confidently incorrect prediction has high loss.
    vector<double> bad_probabilities = {0.9, 0.05, 0.05};

    double bad_loss = crossEntropyLoss(
        bad_probabilities, targets
    );

    CHECK(bad_loss > loss);

    // Test 4: A zero predicted probability must not produce infinity.
    vector<double> zero_probability = {0.0, 1.0, 0.0};

    double zero_loss = crossEntropyLoss(
        zero_probability, targets
    );

    CHECK(isfinite(zero_loss));
    CHECK(zero_loss >= 0.0);

    // Test 5: Soft targets are also supported.
    vector<double> soft_targets = {0.2, 0.5, 0.3};

    double soft_loss = crossEntropyLoss(
        probabilities, soft_targets
    );

    CHECK(isfinite(soft_loss));
    CHECK(soft_loss >= 0.0);

    // Test 6: Average loss over a batch.
    vector<vector<double>> batch_probabilities =
    {
        {0.1, 0.7, 0.2},
        {0.8, 0.1, 0.1}
    };

    vector<vector<double>> batch_targets =
    {
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0}
    };

    double batch_loss = crossEntropyLossBatch(
        batch_probabilities, batch_targets
    );

    double expected_batch_loss =
        (-log(0.7) - log(0.8)) / 2.0;

    CHECK(abs(batch_loss - expected_batch_loss) < 1e-9);

    // Test 7: Reject vectors with different lengths.
    CHECK_THROWS(
        crossEntropyLoss(
            vector<double>{0.5, 0.5},
            vector<double>{1.0}
        )
    );

    // Test 8: Reject probabilities that do not sum to one.
    CHECK_THROWS(
        crossEntropyLoss(
            vector<double>{0.2, 0.2},
            vector<double>{1.0, 0.0}
        )
    );

    // Test 9: Reject targets that do not sum to one.
    CHECK_THROWS(
        crossEntropyLoss(
            vector<double>{0.5, 0.5},
            vector<double>{0.2, 0.2}
        )
    );

    // Test 10: Reject negative probabilities.
    CHECK_THROWS(
        crossEntropyLoss(
            vector<double>{-0.1, 1.1},
            vector<double>{1.0, 0.0}
        )
    );

    // Test 11: Reject invalid epsilon.
    CHECK_THROWS(
        crossEntropyLoss(
            probabilities, targets, 0.0
        )
    );

    if (failures == 0)
    {
        cout << "All cross-entropy tests passed!" << endl;
        return 0;
    }

    cout << failures << " test(s) failed." << endl;
    return 1;
}
