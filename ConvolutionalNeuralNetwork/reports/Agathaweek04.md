# Activation Gradients: ReLU, Sigmoid, Tanh (Week 4)

Author: Agatha
Supervisor: Jeremy
Files: include/activations.hpp, tests/test_activations.cpp
Drawing: class layout photo uploaded with this report

## What it does
During backprop, the gradient from the loss is sent back through every layer.
For an activation y = f(x), the chain rule says:
dL/dx = dL/dy * f'(x)
dL/dy comes from the next layer. backward() multiplies it by f'(x) on each value and returns dL/dx.

## Derivatives

ReLU
1. f(x) = max(0, x)
2. f'(x) = 1 if x > 0
3. f'(x) = 0 if x <= 0 (I use 0 at x = 0)

Sigmoid
1. f(x) = 1 / (1 + e^(-x))
2. f'(x) = e^(-x) / (1 + e^(-x))^2
3. split it: [1 / (1 + e^(-x))] * [e^(-x) / (1 + e^(-x))]
4. = f(x) * (1 - f(x))
5. so f'(x) = y(1 - y), where y is the output
6. largest value is 0.25 at x = 0

Tanh
1. f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
2. f'(x) = 1 - tanh(x)^2
3. so f'(x) = 1 - y^2, where y is the output
4. largest value is 1 at x = 0

## Which value is saved
backward(output, grad_out) takes the output from forward().
- Sigmoid and Tanh: the formulas use y directly
- ReLU: y > 0 exactly when x > 0, so y is enough, and all three have the same signature

## Worked examples
ReLU, x = [-2, 0, 0.5, 3], grad_out = [5, 5, 5, 5]
1. y = [0, 0, 0.5, 3]
2. f' = [0, 0, 1, 1]
3. dL/dx = [5*0, 5*0, 5*1, 5*1]
4. dL/dx = [0, 0, 5, 5]

Sigmoid, x = 0, grad_out = 1
1. y = 0.5
2. f' = 0.5 * (1 - 0.5) = 0.25
3. dL/dx = 1 * 0.25 = 0.25

Tanh, x = 0.5, grad_out = 1
1. y = tanh(0.5) = 0.4621
2. f' = 1 - 0.4621^2 = 1 - 0.2135 = 0.7865
3. dL/dx = 0.7865

## Class layout (namespace act)
ReLU, Sigmoid, Tanh (same layout for all three)
- private: none
- protected: none
- public: forward(const Tensor& input) const : Tensor
- public: backward(const Tensor& output, const Tensor& grad_out) const : Tensor
- Sigmoid and Tanh use the helper functions sigmoid() and tanh_stable() in namespace act

All three use the Tensor class from tensor.hpp.

## Questions and answers
1. What classes did I create? ReLU (new this week), Sigmoid and Tanh (from Week 2, backward finished).
2. Why? Each layer applies one activation to every value, and the three share the same interface, so they are easy to swap in the network.
3. What data do they hold? None. They only work on the tensors passed in.
4. What functions operate on that data?
   - forward(): public, returns f(x) as a new tensor
   - backward(): public, returns dL/dx from the saved output and dL/dy

## OOP concepts

Encapsulation
1. the helper functions sigmoid() and tanh_stable() are kept inside namespace act
2. the classes have no public data, so nothing outside can change their state
3. forward() and backward() are const, they do not change the object
4. outside code only uses forward() and backward()
5. in Week 3 my Softmax class also keeps last_output_ private and gives read access through output()

Inheritance
1. not used in my modules
2. ReLU, Sigmoid and Tanh are separate classes with no base class
3. they share no data, so a base class was not needed
4. in our project, inheritance is used in pooling.hpp: Pooling2D is the base class, and MaxPooling2D and AveragePooling2D are derived from it (Joan's module, not mine)

Polymorphism
1. not used in the run-time sense in my modules (no virtual functions)
2. ReLU, Sigmoid and Tanh have the same function names and signatures, forward(input) and backward(output, grad_out)
3. so one can replace another in the network with only a small code change
4. in pooling.hpp, Pooling2D has a virtual forward() and the derived classes override it (Joan's module)

## Notes
- forward() and backward() return new tensors, the inputs are not changed
- backward() throws std::invalid_argument if output and grad_out have different sizes
- I fixed a typo in the include guard of activations.hpp (#define ACTIATIONS_HPP), because the guard did not protect the file from being included twice

## Testing
test_activations.cpp checks that:
- the Week 2 tests still pass
- ReLU forward turns negatives into 0
- ReLU backward passes the gradient only where x > 0
- sigmoid'(0) = 0.25 and tanh'(0) = 1
- the gradients match a numerical derivative (f(x+h) - f(x-h)) / 2h with h = 1e-6, at six points for ReLU, Sigmoid and Tanh
- the upstream gradient is multiplied (chain rule)
- a size mismatch throws

Compiled with g++ -O3 -Wall -Wextra -pedantic -std=c++17 -Werror, with no warnings.
Also run with AddressSanitizer and UBSan, with no errors.
Result: `All activation tests passed`.
