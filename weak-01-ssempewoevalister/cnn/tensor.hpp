// TEMPORARY STAND-IN for Agatha's Tensor (Week 1 interface standard).
// Used only when CMake option CNN_USE_TENSOR_STUB=ON. Delete once Agatha's PR is merged.
#ifndef CNN_TENSOR_HPP
#define CNN_TENSOR_HPP
#include <cstddef>
#include <vector>
class Tensor {
public:
    std::vector<double> data;
    std::vector<size_t> shape;  // {N, C, H, W}
    Tensor(size_t n, size_t c, size_t h, size_t w)
        : data(n * c * h * w, 0.0), shape{n, c, h, w} {}
    double& operator()(size_t n, size_t c, size_t h, size_t w) {
        return data[((n * shape[1] + c) * shape[2] + h) * shape[3] + w];
    }
    const double& operator()(size_t n, size_t c, size_t h, size_t w) const {
        return data[((n * shape[1] + c) * shape[2] + h) * shape[3] + w];
    }
};
#endif
