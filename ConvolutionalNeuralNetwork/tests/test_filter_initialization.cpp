
#include "filter_initialization.hpp"

#include <iostream>
#include <stdexcept>

using namespace std;

int main()
{
    int failures = 0;

    // Test filter tensor dimensions.
    Tensor filters =
        FilterInitializer::initialize_filters(8, 3, 3, 3);

    if (filters.shape.size() != 4 ||
        filters.shape[0] != 8 ||
        filters.shape[1] != 3 ||
        filters.shape[2] != 3 ||
        filters.shape[3] != 3)
    {
        cerr << "FAILED: Filter tensor dimensions\n";
        failures++;
    }

    // Test that weights are within the requested bounds.
    for (double value : filters.data)
    {
        if (value < -0.1 || value > 0.1)
        {
            cerr << "FAILED: Weight outside initialization bounds\n";
            failures++;
            break;
        }
    }

    // Test bias dimensions and zero initialization.
    Tensor biases = FilterInitializer::initialize_biases(8);

    if (biases.size() != 8)
    {
        cerr << "FAILED: Bias count\n";
        failures++;
    }

    for (double value : biases.data)
    {
        if (value != 0.0)
        {
            cerr << "FAILED: Biases should start at zero\n";
            failures++;
            break;
        }
    }

    // Test invalid dimensions.
    bool rejected = false;

    try
    {
        FilterInitializer::initialize_filters(0, 3, 3, 3);
    }
    catch (const invalid_argument&)
    {
        rejected = true;
    }

    if (!rejected)
    {
        cerr << "FAILED: Zero filter count was not rejected\n";
        failures++;
    }

    // Test invalid distribution bounds.
    rejected = false;

    try
    {
        FilterInitializer::initialize_filters(2, 3, 3, 3, 0.5, -0.5);
    }
    catch (const invalid_argument&)
    {
        rejected = true;
    }

    if (!rejected)
    {
        cerr << "FAILED: Invalid bounds were not rejected\n";
        failures++;
    }

    if (failures == 0)
    {
        cout << "All filter initialization tests passed!" << endl;
        return 0;
    }

    cout << failures << " test(s) failed." << endl;
    return 1;
}
