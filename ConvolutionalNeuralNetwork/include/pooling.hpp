
#ifndef POOLING_HPP
#define POOLING_HPP

#include <cstddef>
#include <stdexcept>
#include "tensor.hpp"

using namespace std;

namespace cnn {

// Parameters used by both max pooling and average pooling.
struct PoolingParams {
    size_t window_h = 2;
    size_t window_w = 2;
    size_t stride_h = 2;
    size_t stride_w = 2;

    // Check that the window and stride sizes are valid.
    void validate() const {
        if (window_h == 0 || window_w == 0) {
            throw invalid_argument(
                "PoolingParams: window size must be > 0"
            );
        }

        if (stride_h == 0 || stride_w == 0) {
            throw invalid_argument(
                "PoolingParams: stride must be > 0"
            );
        }
    }
};

// Calculate one spatial dimension of the pooling output.
inline size_t pooled_dim(
    size_t input,
    size_t window,
    size_t stride
) {
    if (window == 0 || stride == 0) {
        throw invalid_argument(
            "pooled_dim: window and stride must be > 0"
        );
    }

    if (window > input) {
        throw invalid_argument(
            "pooled_dim: window larger than input"
        );
    }

    return (input - window) / stride + 1;
}

// Abstract base class for 2D pooling operations.
class Pooling2D {
public:
    explicit Pooling2D(PoolingParams params);

    virtual ~Pooling2D() = default;

    // Apply the pooling operation to a 4D Tensor.
    virtual ::Tensor forward(const ::Tensor& input) const = 0;

    // Calculate output height.
    size_t output_height(size_t input_h) const {
        return pooled_dim(
            input_h,
            params_.window_h,
            params_.stride_h
        );
    }

    // Calculate output width.
    size_t output_width(size_t input_w) const {
        return pooled_dim(
            input_w,
            params_.window_w,
            params_.stride_w
        );
    }

    // Access pooling parameters.
    const PoolingParams& params() const noexcept {
        return params_;
    }

protected:
    PoolingParams params_;
};

// Max pooling selects the largest value in each window.
class MaxPooling2D : public Pooling2D {
public:
    explicit MaxPooling2D(
        PoolingParams params = PoolingParams{}
    );

    ::Tensor forward(const ::Tensor& input) const override;
};

// Average pooling calculates the mean of each window.
class AveragePooling2D : public Pooling2D {
public:
    explicit AveragePooling2D(
        PoolingParams params = PoolingParams{}
    );

    ::Tensor forward(const ::Tensor& input) const override;
};

} // namespace cnn

#endif // POOLING_HPP
