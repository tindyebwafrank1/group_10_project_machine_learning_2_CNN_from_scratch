#ifndef MAX_POOLING_HPP
#define MAX_POOLING_HPP

#include "tensor.hpp" 
#include <vector>
#include <cstddef>

class MaxPooling {
public:
    size_t pool_height; // Vertical dimension of the pooling window
    size_t pool_width;  // Horizontal dimension of the pooling window
    size_t stride;      // Step traversal rate

    std::vector<size_t> max_indices;

    
    MaxPooling(size_t p_height, size_t p_width, size_t s_stride);

    
    Tensor forward(const Tensor& input);
};

#endif // MAX_POOLING_HPP
