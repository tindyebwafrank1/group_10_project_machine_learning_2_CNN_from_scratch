
#ifndef FILTER_INITIALIZATION_HPP
#define FILTER_INITIALIZATION_HPP

#include "tensor.hpp"

#include <cstddef>

using namespace std;

class FilterInitializer
{
public:
    // Initializes convolution filters using uniform random values.
    // Tensor layout: (num_filters, channels, kernel_height, kernel_width).
    static Tensor initialize_filters(
        size_t num_filters,
        size_t channels,
        size_t kernel_height,
        size_t kernel_width,
        double min_val = -0.1,
        double max_val = 0.1
    );

    // Creates one zero-initialized bias per filter.
    // Tensor layout: (1, num_filters, 1, 1).
    static Tensor initialize_biases(size_t num_filters);
};

#endif
