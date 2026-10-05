#ifndef CNN_POOLING_HPP
#define CNN_POOLING_HPP

#include <cstddef>
#include <stdexcept>

namespace cnn {

// Placeholder: replace with #include "cnn/tensor.hpp" once the
// tensor owner has committed their header.
class Tensor;

// Window size and stride for a 2D pooling layer.
// Defaults give the common 2x2 window with stride 2.
struct PoolingParams {
    std::size_t window_h = 2;
    std::size_t window_w = 2;
    std::size_t stride_h = 2;
    std::size_t stride_w = 2;

    // Throws std::invalid_argument if any value is zero.
    void validate() const {
        if (window_h == 0 || window_w == 0)
            throw std::invalid_argument("PoolingParams: window size must be > 0");
        if (stride_h == 0 || stride_w == 0)
            throw std::invalid_argument("PoolingParams: stride must be > 0");
    }
};

}  // namespace cnn

#endif  // CNN_POOLING_HPP