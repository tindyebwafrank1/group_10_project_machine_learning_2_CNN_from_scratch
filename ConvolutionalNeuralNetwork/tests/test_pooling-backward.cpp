#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "tensor.hpp"
#include "pooling_backward.hpp"

using namespace std;

int main() {
    cout << "[RUNNING] Verifying Pooling Backward Pass Routines..." << endl;

    vector<size_t> input_shape = {1, 1, 4, 4};
    Tensor dL_dy(1, 1, 2, 2);

    dL_dy(0, 0, 0, 0) = 2.0;
    dL_dy(0, 0, 0, 1) = 4.0;
    dL_dy(0, 0, 1, 0) = 6.0;
    dL_dy(0, 0, 1, 1) = 8.0;

    PoolingBackward pool_back(2, 2, 2);
    pool_back.max_indices = {5, 3, 8, 15};

    Tensor dL_dx_max = pool_back.max_pooling_backward(dL_dy, input_shape);

    assert(dL_dx_max.size() == 16);
    assert(abs(dL_dx_max.data[5] - 2.0) < 1e-6);
    assert(abs(dL_dx_max.data[3] - 4.0) < 1e-6);
    assert(abs(dL_dx_max.data[8] - 6.0) < 1e-6);
    assert(abs(dL_dx_max.data[15] - 8.0) < 1e-6);
    assert(abs(dL_dx_max.data[0]) < 1e-6);
    assert(abs(dL_dx_max.data[1]) < 1e-6);

    cout << " -> Pass: Max pooling backward pass routes gradients via argmax indices." << endl;

    Tensor dL_dx_avg = pool_back.average_pooling_backward(dL_dy, input_shape);

    assert(abs(dL_dx_avg(0, 0, 0, 0) - 0.5) < 1e-6);
    assert(abs(dL_dx_avg(0, 0, 0, 1) - 0.5) < 1e-6);
    assert(abs(dL_dx_avg(0, 0, 1, 0) - 0.5) < 1e-6);
    assert(abs(dL_dx_avg(0, 0, 1, 1) - 0.5) < 1e-6);
    assert(abs(dL_dx_avg(0, 0, 2, 2) - 2.0) < 1e-6);
    assert(abs(dL_dx_avg(0, 0, 2, 3) - 2.0) < 1e-6);
    assert(abs(dL_dx_avg(0, 0, 3, 2) - 2.0) < 1e-6);
    assert(abs(dL_dx_avg(0, 0, 3, 3) - 2.0) < 1e-6);

    cout << " -> Pass: Average pooling backward pass distributes gradients uniformly." << endl;

    cout << "\n[SUCCESS] Mugarura (M5) Week 4 backpropagation routines verified successfully!" << endl;
    return 0;
}