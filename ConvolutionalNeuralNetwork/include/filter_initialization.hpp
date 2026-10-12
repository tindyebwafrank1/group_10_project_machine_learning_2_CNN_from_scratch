#ifndef FILTER_INITIALIZER_HPP
#define FILTER_INITIALIZER_HPP

#include "tensor.hpp" 
#include <random>
#include <cstddef>

class FilterInitializer {
public:
    /**
     
     * @param num_filters (K)   - Number of unique target filters (output channels).
     * @param channels (C)      - Number of input channels matching the upstream tensor layout.
     * @param kernel_height (Kh)- Vertical dimension of the localized spatial filter.
     * @param kernel_width (Kw) - Horizontal dimension of the localized spatial filter.
     * @param min_val           - Lower distribution bound for uniform random setup.
     * @param max_val           - Upper distribution bound for uniform random setup.

     */
    static Tensor initialize_filters(size_t num_filters, size_t channels, 
                                     size_t kernel_height, size_t kernel_width,
                                     double min_val = -0.1, double max_val = 0.1) {
        
        // 1. Instantiate tensor template layout: {N=K, C=C, H=Kh, W=Kw}
    
        Tensor filter_tensor(num_filters, channels, kernel_height, kernel_width);
        
        std::mt19937 rng(42); 
        std::uniform_real_distribution<double> distribution(min_val, max_val);
        
        
        for (size_t i = 0; i < filter_tensor.size(); ++i) {
            filter_tensor.data[i] = distribution(rng);
        }
        
        return filter_tensor;
    }

    static Tensor initialize_biases(size_t num_filters) {

        return Tensor(1, num_filters, 1, 1);
    }
};

#endif // FILTER_INITIALIZER_HPP
