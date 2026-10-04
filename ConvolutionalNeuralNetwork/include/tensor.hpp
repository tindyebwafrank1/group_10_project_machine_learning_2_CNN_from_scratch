#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>

// Tensor stores 4D data (N, C, H, W) in one flat block of memory
class Tensor
{
public:
    std::vector<double> data; // flat storage of all values
    std::vector<size_t> shape; // {N, C, H, W}

    Tensor();
    Tensor(size_t n, size_t c, size_t h, size_t w);

    size_t size() const;

    // access one value using (n, c, h, w)
    double& operator()(size_t n, size_t c, size_t h, size_t w);
    const double& operator()(size_t n, size_t c, size_t h, size_t w) const;

    void fill(double value);
    void reshape(size_t n, size_t c, size_t h, size_t w);

    // basic operations (return a new Tensor)
    Tensor add(const Tensor& other) const;
    Tensor subtract(const Tensor& other) const;
    Tensor multiply(const Tensor& other) const; // element by element
    Tensor scale(double factor) const;
};

#endif