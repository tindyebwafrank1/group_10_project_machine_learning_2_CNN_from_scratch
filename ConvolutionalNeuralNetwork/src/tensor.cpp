#include "tensor.hpp"
#include <stdexcept>
using namespace std;

// default constructor: empty tensor
Tensor::Tensor()
{
    shape = {0, 0, 0, 0};
}

// creates a tensor of size n*c*h*w filled with zeros
Tensor::Tensor(size_t n, size_t c, size_t h, size_t w)
{
    shape = {n, c, h, w};
    data.assign(n * c * h * w, 0.0);
}

// total number of values
size_t Tensor::size() const
{
    return data.size();
}

// index = n*(C*H*W) + c*(H*W) + h*W + w
double& Tensor::operator()(size_t n, size_t c, size_t h, size_t w)
{
    return data[n * (shape[1] * shape[2] * shape[3])
              + c * (shape[2] * shape[3])
              + h * shape[3]
              + w];
}

// same as above but for const tensors (read only)
const double& Tensor::operator()(size_t n, size_t c, size_t h, size_t w) const
{
    return data[n * (shape[1] * shape[2] * shape[3])
              + c * (shape[2] * shape[3])
              + h * shape[3]
              + w];
}

// set every value to the same number
void Tensor::fill(double value)
{
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = value;
    }
}

// change the shape, the total size must stay the same
void Tensor::reshape(size_t n, size_t c, size_t h, size_t w)
{
    if (n * c * h * w != data.size())
    {
        throw invalid_argument("reshape: total size must stay the same");
    }
    shape = {n, c, h, w};
}

// element by element addition
Tensor Tensor::add(const Tensor& other) const
{
    if (shape != other.shape)
    {
        throw invalid_argument("add: shapes do not match");
    }
    Tensor result(shape[0], shape[1], shape[2], shape[3]);
    for (size_t i = 0; i < data.size(); i++)
    {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

// element by element subtraction
Tensor Tensor::subtract(const Tensor& other) const
{
    if (shape != other.shape)
    {
        throw invalid_argument("subtract: shapes do not match");
    }
    Tensor result(shape[0], shape[1], shape[2], shape[3]);
    for (size_t i = 0; i < data.size(); i++)
    {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

// element by element multiplication
Tensor Tensor::multiply(const Tensor& other) const
{
    if (shape != other.shape)
    {
        throw invalid_argument("multiply: shapes do not match");
    }
    Tensor result(shape[0], shape[1], shape[2], shape[3]);
    for (size_t i = 0; i < data.size(); i++)
    {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

// multiply every value by one number
Tensor Tensor::scale(double factor) const
{
    Tensor result(shape[0], shape[1], shape[2], shape[3]);
    for (size_t i = 0; i < data.size(); i++)
    {
        result.data[i] = data[i] * factor;
    }
    return result;
}
