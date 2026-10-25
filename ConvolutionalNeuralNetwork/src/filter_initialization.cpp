
#include "filter_initialization.hpp"

#include <random>
#include <stdexcept>

using namespace std;

Tensor FilterInitializer::initialize_filters(
    size_t num_filters,
    size_t channels,
    size_t kernel_height,
    size_t kernel_width,
    double min_val,
    double max_val)
{
    // Validate the filter dimensions.
    if (num_filters == 0 || channels == 0 ||
        kernel_height == 0 || kernel_width == 0)
    {
        throw invalid_argument(
            "Filter dimensions must all be greater than zero."
        );
    }

    // Validate the uniform distribution bounds.
    if (min_val > max_val)
    {
        throw invalid_argument(
            "Minimum value cannot exceed maximum value."
        );
    }

    // Create the filter tensor in NCHW layout.
    Tensor filter_tensor(
        num_filters,
        channels,
        kernel_height,
        kernel_width
    );

    // Fixed seed gives reproducible initialization.
    mt19937 rng(42);
    uniform_real_distribution<double> distribution(
        min_val,
        max_val
    );

    // Initialize every filter weight.
    for (size_t i = 0; i < filter_tensor.size(); ++i)
    {
        filter_tensor.data[i] = distribution(rng);
    }

    return filter_tensor;
}

Tensor FilterInitializer::initialize_biases(size_t num_filters)
{
    if (num_filters == 0)
    {
        throw invalid_argument(
            "Number of filters must be greater than zero."
        );
    }

    // One zero-initialized bias per filter.
    Tensor biases(1, num_filters, 1, 1);
    biases.fill(0.0);

    return biases;
}
