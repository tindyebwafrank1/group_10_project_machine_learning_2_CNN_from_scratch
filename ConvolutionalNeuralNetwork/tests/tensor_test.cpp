#include <iostream>
#include "tensor.hpp"
using namespace std;

int main()
{
    // create a 1x1x3x3 tensor
    Tensor t(1, 1, 3, 3);
    cout << "size: " << t.size() << endl;

    // set and read one value
    t(0, 0, 1, 1) = 5.0;
    cout << "value: " << t(0, 0, 1, 1) << endl;

    // fill with 2
    t.fill(2.0);
    cout << "after fill: " << t(0, 0, 2, 2) << endl;

    // add, multiply and scale
    Tensor a(1, 1, 2, 2);
    Tensor b(1, 1, 2, 2);
    a.fill(3.0);
    b.fill(4.0);
    cout << "add: " << a.add(b)(0, 0, 0, 0) << endl;
    cout << "subtract: " << b.subtract(a)(0, 0, 0, 0) << endl;
    cout << "multiply: " << a.multiply(b)(0, 0, 0, 0) << endl;
    cout << "scale: " << a.scale(2.0)(0, 0, 0, 0) << endl;

    // reshape
    t.reshape(1, 1, 1, 9);
    cout << "reshaped width: " << t.shape[3] << endl;

    return 0;
}
