
#ifndef CNN_FLATTEN_HPP
#define CNN_FLATTEN_HPP

#include <vector>
#include "tensor.hpp"

using namespace std;

namespace cnn {

vector<vector<double>> flatten(const ::Tensor& input);

} // namespace cnn

#endif // CNN_FLATTEN_HPP
