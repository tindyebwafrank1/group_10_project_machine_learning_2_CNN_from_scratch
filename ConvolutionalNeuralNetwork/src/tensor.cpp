
#include "tensor.hpp"
#include <stdexcept>

using namespace std;

// Default constructor: creates an empty tensor.
Tensor::Tensor()
{
    shape = {0, 0, 0, 0};
}

// Creates a tensor with dimensions N, C, H, W,
// initialized with zeros.
Tensor::Tensor(size_t n, size_t c, size_t h, size_t w)
{
    shape = {n, c, h, w};
    data.assign(n * c * h * w, 0.0);
}

// Returns the total number of elements.
size_t Tensor::size() const
{
    return data.size();
}

// Access an element using (n, c, h, w).
double& Tensor::operator()(
    size_t n, size_t c, size_t h, size_t w
)
{
    if (shape.size() != 4 ||
        n >= shape[0] ||
        c >= shape[1] ||
        h >= shape[2] ||
        w >= shape[3])
    {
        throw out_of_range("Tensor index out of range");
    }

    // Flat index for contiguous NCHW storage.
    size_t index = n * (shape[1] * shape[2] * shape[3])
                 + c * (shape[2] * shape[3])
                 + h * shape[3]
                 + w;

    return data[index];
}

// Read-only access for const tensors.
const double& Tensor::operator()(
    size_t n, size_t c, size_t h, size_t w
) const
{
    if (shape.size() != 4 ||
        n >= shape[0] ||
        c >= shape[1] ||
        h >= shape[2] ||
        w >= shape[3])
    {
        throw out_of_range("Tensor index out of range");
    }

    size_t index = n * (shape[1] * shape[2] * shape[3])
                 + c * (shape[2] * shape[3])
                 + h * shape[3]
                 + w;

    return data[index];
}

// Sets every element to the same value.
void Tensor::fill(double value)
{
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = value;
    }
}

// Changes the tensor dimensions without changing its size.
void Tensor::reshape(size_t n, size_t c, size_t h, size_t w)
{
    if (n * c * h * w != data.size())
    {
        throw invalid_argument(
            "reshape: total size must stay the same"
        );
    }

    shape = {n, c, h, w};
}

// Element-wise addition.
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

// Element-wise subtraction.
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

// Element-wise multiplication.
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

// Multiplies every element by a scalar.
Tensor Tensor::scale(double factor) const
{
    Tensor result(shape[0], shape[1], shape[2], shape[3]);

    for (size_t i = 0; i < data.size(); i++)
    {
        result.data[i] = data[i] * factor;
    }

    return result;
}
