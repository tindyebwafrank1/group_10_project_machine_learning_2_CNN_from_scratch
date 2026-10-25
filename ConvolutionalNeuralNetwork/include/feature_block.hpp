
#ifndef CNN_FEATURE_BLOCK_HPP
#define CNN_FEATURE_BLOCK_HPP

#include "tensor.hpp"

#include <cstddef>

using namespace std;

namespace cnn {

class FeatureBlock {
private:
    Tensor filters;

    size_t pool_window;
    size_t pool_stride;

public:
    FeatureBlock(
        const Tensor& convolution_filters,
        size_t pooling_window = 2,
        size_t pooling_stride = 2
    );

    Tensor forward(const Tensor& input) const;
};

} 

#endif
