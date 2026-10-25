# SGD Optimizer Module (`cnn_library/optimizer.hpp`)

Owner:  Evalister | Reviewer: Joan  | Header-only, namespace `CNN`

## Purpose
Applies the Week 4 update rule W <- W - eta * grad(W) to any layer's parameters
(conv kernels, fully connected weights and biases).

## Usage
```cpp
#include "cnn_library/optimizer.hpp"
CNN::SGD opt(0.01);                       // learning rate only: plain SGD
CNN::SGD opt2(0.01, 0.9, 1e-4);           // + momentum and L2 weight decay

// after the backward pass, for each parameter and its gradient:
opt.step(fc_weights, fc_weight_grads);    // std::vector<double> or Tensor
opt.step(conv_kernel, conv_kernel_grads, 1.0 / batch_size); // if grads were summed
CNN::zero_gradients(fc_weight_grads);     // clear before the next mini-batch
opt.decay_learning_rate(0.95);            // optional, once per epoch
```

## API
| Member | Description |
|---|---|
| `SGD(lr, momentum=0, weight_decay=0)` | lr > 0, momentum in [0,1), weight_decay >= 0 |
| `step(vector<double>& p, const vector<double>& g, scale=1.0)` | in-place update |
| `step(Tensor& p, const Tensor& g, scale=1.0)` | same, shapes must match |
| `set_learning_rate`, `decay_learning_rate(f)` | learning-rate control |
| `reset_state()` | clear momentum state |
| `zero_gradients(...)` | zero a gradient vector or Tensor |

## Formulas
- Plain: W <- W - lr * (scale * grad + weight_decay * W)
- Momentum: v <- mu * v - lr * g_eff ; W <- W + v

## Design decisions
- Plain SGD is the default; momentum and weight decay are opt-in and off unless asked for.
- `grad_scale` lets the training loop (M8) pass summed or averaged batch gradients without a copy.
- Momentum state is stored per parameter, keyed by the parameter's address, so one optimizer can serve every layer.
- Bad input (size/shape mismatch, NaN or infinite gradient) throws before any parameter changes.

## Integration (who calls what)
- Inputs come from the backward passes: Peter (FC gradients), Joan (Conv2D kernel gradients), Frank (loss/softmax).
- Edrine (M8) calls `step` for every parameter inside the mini-batch loop.

## Limitations
- Momentum state is keyed by parameter address: parameters must stay at a fixed address and size for the whole run (e.g. do not store them in a `std::vector` that grows after training starts), otherwise call `reset_state()`.
- Only SGD (with optional momentum and L2). No Adam or RMSProp.
- Not thread safe.
