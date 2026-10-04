#include <iostream>
#include "tensor.h"

int main()
{
    Tensor t({1, 1, 3, 3});

    std::cout << "size: " << t.size() << std::endl;

    t(0, 0, 1, 1) = 5.0f;
    std::cout << "value: " << t(0, 0, 1, 1) << std::endl;

    t.fill(2.0f);
    std::cout << "after fill: " << t(0, 0, 2, 2) << std::endl;

    t.reshape({1, 1, 1, 9});
    std::cout << "reshaped width: " << t.getShape()[3] << std::endl;

    return 0;
}