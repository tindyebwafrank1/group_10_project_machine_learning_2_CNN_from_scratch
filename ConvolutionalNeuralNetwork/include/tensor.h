#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

class Tensor
{
private:
    std::vector<int> shape;
    std::vector<float> data;

public:
    Tensor();
    Tensor(const std::vector<int>& shape);

    int size() const;
    const std::vector<int>& getShape() const;

    float& operator()(int n, int c, int h, int w);
    float operator()(int n, int c, int h, int w) const;

    void fill(float value);
    void reshape(const std::vector<int>& newShape);
};

#endif
