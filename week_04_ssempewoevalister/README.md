# CNN Library from Scratch in C++17

A lightweight, header-only Convolutional Neural Network (CNN) library implemented entirely in C++17. This repository contains the complete modular pipeline for building, evaluating, and optimizing deep learning models from scratch without external linear algebra dependencies.


## 1. Project Name and Description
**CNN Library from Scratch in C++17**
This project is a modular, header-only deep learning library developed from first principles. It provides low-level implementations of fundamental neural network layers, execution workflows, structural evaluation metrics, and core optimization modules designed to build and train models directly from source.

## 2. Objectives
* **Low-Level Mastery:** Build deep learning components directly without relying on external acceleration or linear algebra toolkits.
* **Modular Pipeline Construction:** Assemble an end-to-end framework spanning forward/backward convolution passes, layer orchestration block groupings, loss calculation, evaluation matrix tracking, and parametric weight optimization.
* **Execution Safety:** Implement extensive fail-fast parameter validation to capture structural anomalies, sizing mismatches, or divergent data tracking before layers carry out destructive updates.

## 3. Main Features
* **Zero Dependencies:** Header-only layout designed for single-command builds alongside your `main.cpp` entry driver.
* **Modular Feature Layer Blocks:** Chainable structures merging 2D Convolution layers, downstream non-linear Activations, and spatial Pooling mechanics into single execution segments.
* **Granular Metrics Evaluation Engine:** Comprehensive performance tracking (`metrics.hpp`) offering class-by-class confusion matrix telemetry, precision profiles, recall trends, and F1 calculations.
* **Production-Grade SGD Optimizer:** High-fidelity optimizer incorporating standard gradient updates, optional parameter-bound momentum tracking, L2 weight decay penalties, and configurable gradient scaling controls.

## 4. Project Structure
```text
.
├── CMakeLists.txt
├── examples/
│   └── sgd_example.cpp
├── include/
│   └── cnn_library/
│       ├── optimizer.hpp
│       └── tensor.hpp
├── standins/
│   └── include/
│       └── tensor.hpp
└── tests/
    └── test_optimizer.cpp
```

## 5. Requirements
* **Compiler Framework:** `g++` or `clang++` toolchains matching full C++17 language compliance standards.
* **Build System:** CMake suite version 3.10 or higher.
* **Runtime Environments:** Clean compilation profiles verified under active `AddressSanitizer (ASan)` and `UndefinedBehaviorSanitizer (UBSan)` diagnostic passes.

## 6. Build Instructions
To compile the bundled testing targets and programmatic examples using CMake, open your workspace terminal and execute:
```bash
mkdir build && cd build && cmake .. && make
```
To explicitly compile a standalone application binary direct via `g++` with high compiler warning diagnostic flags and aggressive optimization options, execute:
```bash
g++ -O3 -Wall -Wextra -std=c++17 main.cpp -o cnn_app
```

## 7. How to Use the Library
The following code snippet demonstrates how to instantiate the optimizer suite to perform parameter tracking and clean local accumulation blocks:

```cpp
#include "cnn_library/optimizer.hpp"

// Setup your optimizer with a baseline learning rate, momentum, and L2 regularization
CNN::SGD optimizer(0.01, 0.9, 1e-4);

// Within your mini-batch execution loop:
// Pass your parameter array, corresponding gradient array, and an average-scaling factor
optimizer.step(layer_weights, weight_gradients, 1.0 / batch_size);

// Reset gradient tracking matrices before initializing the subsequent mini-batch pass
CNN::zero_gradients(weight_gradients);
```

## 8. Examples
* **`examples/sgd_example.cpp`:** Implements a localized mini-batch optimization workflow to fit an abstract linear regression curve ($y = 2x + 1$) using random dataset shuffling, forward prediction passes, backpropagated loss derivation, and progressive convergence evaluation.

## 9. How to Run Tests
To evaluate compilation integrity and math assertions via CMake, run:
```bash
ctest
```
To build and execute testing suites manually to review raw console logs, compile and run the test source directly:
```bash
g++ -Wall -Wextra -std=c++17 tests/test_optimizer.cpp -Iinclude -Istandins/include -o test_optimizer
./test_optimizer
```

## 10. Limitations
* **Address Pointer Consistency:** The internal velocity map relies on tracking specific, unchanging raw memory addresses; do not push parameters into dynamically expanding containers that trigger array relocations without calling `reset_state()`.
* **Algorithmic Scope Restrictions:** Built specifically around specialized variants of Stochastic Gradient Descent; alternative decoupled adaptive rate algorithms (such as Adam, RMSProp, or AdaGrad) are not natively supported.
* **Concurrency Limitations:** The core calculation architecture is strictly optimized for single-threaded tasks and is not explicitly thread-safe.

## 11. Contributions by Member
**Evalister:** Designed and implemented the multi-channel and multi-filter forward convolution pass frameworks (Week 1); built out the pipeline integration for structural layer chaining within `feature_block.hpp` (Week 2); developed the multi-class Confusion Matrix layout and performance equations in `metrics.hpp` (Week 3); authored the complete state-tracking `SGD` class infrastructure inside `optimizer.hpp` (Week 4).

## 12. Collaborative Workflow
Our technical workflow followed a synchronized development loop mapping progressively from Member 1through Member 8. Core module assignments were developed using specialized, isolated Git branches. Updates were proposed through formal Pull Requests (PRs), requiring complete peer review and code validation inspections before final trunk integrations were safely merged into our main branch architecture.