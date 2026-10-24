#ifndef TENSOR_HPP
#define TENSOR_HPP
// STAND-IN for Agatha's Tensor, matching the Week 1 interface standard.
// Not part of Evalister's deliverable: delete once the real tensor.hpp is on main.
#include <cstddef>
#include <vector>
namespace CNN {
    class Tensor {
    public:
        std::vector<double> data;
        std::vector<size_t> shape; // {N, C, H, W}
        Tensor(size_t n, size_t c, size_t h, size_t w)
            : data(n * c * h * w, 0.0), shape{n, c, h, w} {}
        double& operator()(size_t n, size_t c, size_t h, size_t w) {
            return data[n * (shape[1] * shape[2] * shape[3]) + c * (shape[2] * shape[3]) + h * shape[3] + w];
        }
        const double& operator()(size_t n, size_t c, size_t h, size_t w) const {
            return data[n * (shape[1] * shape[2] * shape[3]) + c * (shape[2] * shape[3]) + h * shape[3] + w];
        }
    };
}
#endif
