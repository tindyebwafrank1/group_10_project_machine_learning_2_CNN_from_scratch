#include "tensor.h"
#include <stdexcept>

Tensor::Tensor()
{
}

Tensor::Tensor(const std::vector<int>& shape)
{
    this->shape = shape;
    int total = 1;
    for (size_t i = 0; i < shape.size(); i++)
    {
        total *= shape[i];
    }
    data.assign(total, 0.0f);
}

int Tensor::size() const
{
    return data.size();
}

const std::vector<int>& Tensor::getShape() const
{
    return shape;
}

float& Tensor::operator()(int n, int c, int h, int w)
{
    return data[((n * shape[1] + c) * shape[2] + h) * shape[3] + w];
}

float Tensor::operator()(int n, int c, int h, int w) const
{
    return data[((n * shape[1] + c) * shape[2] + h) * shape[3] + w];
}

void Tensor::fill(float value)
{
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = value;
    }
}

void Tensor::reshape(const std::vector<int>& newShape)
{
    int total = 1;
    for (size_t i = 0; i < newShape.size(); i++)
    {
        total *= newShape[i];
    }
    if (total != size())
    {
        throw std::invalid_argument("reshape: size mismatch");
    }
    shape = newShape;
}
