# Tensor Class (tensor.h), Week 1

Tensor stores 4D data {N,C,H,W} as doubles in one flat vector.
Index(n,c,h,w) = n*(C*H*W) + c*(H*W) + h*W + w.
Size mismatches throw std::invalid_argument.

| Function | Input -> Output | Behaviour |
|---|---|---|
| Tensor(n,c,h,w) | 4 sizes -> Tensor | makes a tensor filled with zeros |
| operator()(n,c,h,w) | 4 indices -> double& | reads or writes one value |
| size() | none -> size_t | total number of values |
| fill(value) | double -> void | sets every value to value |
| reshape(n,c,h,w) | 4 sizes -> void | changes shape, total size must stay the same |
| add(other) | Tensor -> Tensor | element by element sum, shapes must match |
| subtract(other) | Tensor -> Tensor | element by element difference, shapes must match |
| multiply(other) | Tensor -> Tensor | element by element product, shapes must match |
| scale(factor) | double -> Tensor | multiplies every value by factor |

Tested in tests/tensor_test.cpp: size, indexing, fill, add, subtract, multiply, scale and reshape all give the expected values.
