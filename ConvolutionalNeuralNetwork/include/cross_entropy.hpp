#ifndef CROSS_ENTROPY_HPP
#define CROSS_ENTROPY_HPP

#include "tensor.hpp" 
#include <cstddef>

class CrossEntropyLoss {
private:
    
    const double epsilon = 1e-15; 

public:
    CrossEntropyLoss() = default;

    
    double forward(const Tensor& predictions, const Tensor& targets) const;
};

#endif 
