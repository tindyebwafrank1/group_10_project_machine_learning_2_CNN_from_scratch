# Week 4 Progress Report

## Completed
- Implemented the `SGD` optimizer in `include/cnn_library/optimizer.hpp`: W <- W - lr * grad, for `std::vector<double>` and `Tensor`.
- Optional momentum, L2 weight decay, learning-rate decay, `grad_scale` for summed vs averaged batch gradients, and `zero_gradients` helper.
- Input validation: bad learning rate, size/shape mismatch, NaN/infinite gradients (checked before any update).
- Tests in `tests/test_optimizer.cpp` (hand-computed updates, momentum over two steps, per-parameter state, convergence on a quadratic, invalid input, no partial update); clean under `-Wall -Wextra` and ASan/UBSan.
- Example `examples/sgd_example.cpp` (mini-batch fit of y = 2x + 1, converges to w=2, b=1); module doc `docs/sgd.md`.

## In Progress
- PR to main for review by Joan.
- Agreeing with Edrine on how the training loop calls `step` for every layer's parameters.
- Replacing the temporary `standins/tensor.hpp` with M4's real Tensor.

## Challenges
 Late submissions making reviewing of work hard
 Depends on the backward passes from Peter, Joan, Frank and others producing gradients in the same shape as their parameters.

## Next Week
- Final release: help verify the full build with `g++ -O3 -Wall -Wextra -std=c++17 main.cpp -o cnn_app`, and test accuracy logs.
- Final report and presentation preparation.

## AI Use
- Tool: Claude
- Purpose: Generated the first draft of the SGD module
- Reason: Speed up the initial draft. I reviewed and ran the code and can explain it. 
