# Sigmoid & Tanh Activation Modules (Week 2)

**Author:** Agatha (M4)
**Supervisor:** Jeremy (M3)
**Files:** `include/activations.hpp`, `src/test_activations.cpp`

## What it does
Sigmoid and Tanh are activation functions that add non-linearity to the CNN. Each one is applied to every value in a Tensor separately, so the output has the same shape as the input.

## Formulas
- **Sigmoid:** 1 / (1 + e^(-x)), with output range (0, 1)
- **Tanh:** (e^x - e^(-x)) / (e^x + e^(-x)), with output range (-1, 1)

Both curves are S-shaped. Tanh is centred on 0, while sigmoid is centred on 0.5. For large positive or negative inputs both flatten out (saturate).

## Stability
- **Sigmoid:** the direct formula overflows for large negative x because e^(-x) becomes huge. The code uses `1/(1+e^-x)` for x >= 0 and `e^x/(1+e^x)` for x < 0, so `exp()` is never called on a large positive number.
- **Tanh:** the textbook formula gives NaN for very large inputs. The code uses `std::tanh`, which safely returns +1 or -1.

## Usage
```cpp
act::Sigmoid sig;
Tensor y = sig.forward(x);
```
Both classes also have a `backward()` function for Week 4, using sigmoid' = y(1-y) and tanh' = 1-y^2.

## Testing
`test_activations.cpp` checks that:
- outputs stay finite for inputs of -1000 to 1000
- outputs stay within range
- sigmoid(0) = 0.5 and tanh(0) = 0
- the shape is preserved
- backward() matches a numerical derivative

Result: `All activation tests passed`.