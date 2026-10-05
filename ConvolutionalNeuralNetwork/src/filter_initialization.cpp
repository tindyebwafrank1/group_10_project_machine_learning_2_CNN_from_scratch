#include <filter_initialization.h>
#include <random>
#include <tensor.hpp>

Tensor FilterInitializer::initialize_filters(size_t num_filters, size_t channels, 
                                             size_t kernel_height, size_t kernel_width,
                                             double min_val, double max_val) 
{
    Tensor filter_tensor(num_filters, channels, kernel_height, kernel_width);
    
    std::mt19937 rng(42); 
    std::uniform_real_distribution<double> distribution(min_val, max_val);
    
    for (size_t i = 0; i < filter_tensor.size(); ++i) 
    {
        filter_tensor.data[i] = distribution(rng);
    }
    
    return filter_tensor;
}

Tensor FilterInitializer::initialize_biases(size_t num_filters) 
{
    return Tensor(1, num_filters, 1, 1);
}
