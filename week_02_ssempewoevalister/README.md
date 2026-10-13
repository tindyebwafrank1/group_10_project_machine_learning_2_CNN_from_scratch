# CNN From Scratch in C++: Week 2 Feature Extraction Pipeline

This directory contains the production-grade implementation, verification test suites, and documentation tools for **Module M7: The Feature-Extraction Block Builder (`FeatureBlock`)**. 

Instead of treating convolutional mechanics as disjoint computational operations, this engine serves as the **high-level structural wrapper** for our network's feedforward system—sequentially binding a multi-channel/multi-filter 2D convolution layer, a non-linear activation element, and a spatial downsampling pooling block into an optimized, encapsulated forward execution sequence.

---

## 📁 Repository Directory Structure

The files are organized to support self-contained modular builds using local workspace dependency stand-ins until adjacent teammate integration branches are systematically merged into `main`:

```text
group_10_project_machine_learning_2_CNN_from_scratch/
├── CMakeLists.txt                 # Master CMake build configuration framework
├── include/                       # Core production interface headers
│   └── cnn_library/
│       └── feature_block.hpp      # Architectural core: FeatureBlock & FeatureExtractor
├── standins/                      # Isolation directory containing team component stubs
│   └── include/
│       └── cnn_library/
│           ├── activations.hpp    # Mock ReLU, Sigmoid, and Tanh abstractions
│           ├── layers.hpp         # Stand-in batch/stride/padding 2D Conv layer
│           ├── pooling.hpp        # Spatial reduction engine (Max/Average pooling)
│           └── tensor.hpp         # Contiguous NCHW data storage vector abstraction
├── tests/
│   └── test_feature_block.cpp     # Deterministic plain-assert unit validation tests
└── examples/
    └── feature_block_example.cpp  # Synthetic 28x28 grayscale image execution simulation
```

---

## ⚙️ Compilation & Testing Pipeline

Our build configuration leverages `CMake` to manage target definitions, linking rules, and safety flags without requiring manual source-string tracking.

Execute the following commands in your VS Code terminal workspace to compile the components and run the test harness:

```bash
# 1. Generate the isolated CMake build workspace structure
cmake -S . -B build

# 2. Compile the feature block test and simulation targets
cmake --build build

# 3. Execute the automated validation verification test suite
./build/test_feature_block
```

### Expected Output Logs
Upon a successful execution path, the evaluation matrix processes all batch variations, channel combinations, and 15 distinct structural layer configurations to output:
```text
All feature block tests passed.
```

### Running the Visual Processing Simulation
To trace the operational path of a mock grayscale image tensor vector (\(1 \times 1 \times 28 \times 28\)) downsampled through a 3-block pipeline down to an exact feature map state, execute:
```bash
./build/feature_block_example
```

---

## 🧠 Architectural Architecture & Mathematical Flow

The forward logic within `FeatureBlock::forward` executes three consecutive computational phases:

```text
  [Input Tensor] 
        │
        ▼
 ┌──────────────┐
 │    Conv2D    │  ──► Validates shape bounds & sums weights over input channels
 └──────────────┘
        │
        ▼
 ┌──────────────┐
 │  Activation  │  ──► Maps feature data arrays through ReLU, Sigmoid, or Tanh
 └──────────────┘
        │
        ▼
 ┌──────────────┐
 │   Pooling    │  ──► Downsamples spatial grids via Max or Average windows
 └──────────────┘
        │
        ▼
 [Feature Tensors]
```

### Dimensional Sizing Equations
To enforce calculation safety, spatial boundaries are evaluated inside `output_shape(H, W)` before memory tracking allocations happen, verifying dimensions via the integer floor formulas:

*   **Convolution Matrix Limits**: 
    \[H_{conv} = \left\lfloor \frac{H_{in} - K + 2P}{S} \right\rfloor + 1\]
*   **Spatial Pooling Limits**: 
    \[H_{pool} = \left\lfloor \frac{H_{conv} - \text{window}}{\text{pool\_stride}} \right\rfloor + 1\]

### Dynamic Filter Initialization
Weight fields within the kernel matrix match **He-style Normal Initialization** rules to minimize gradient exploding/vanishing anomalies:
\[\sigma = \sqrt{\frac{2}{\text{in\_channels} \times K \times K}}\]
Tensors are automatically assigned deterministic tracking seeds based on block index locations to guarantee reproducible validations.

---

## 🚀 Forward Roadmap Adjustments

1. **Week 3 Integration (Stateful Gradient Buffers)**: The current Week 2 iteration operates under strict `const` optimization targets because it behaves as an stateless inference block. For Week 3, the `FeatureBlock` layer will be refactored to preserve inner forward intermediate output buffers (`z` and `a`), allowing the system to retroactively process backpropagation loss matrix operations (\(\frac{\partial L}{\partial X}\) and \(\frac{\partial L}{\partial W}\)).
2. **Week 4 Optimization (Parallel Vectorization)**: Nested iteration layers inside `layers.hpp` and `pooling.hpp` checking elements pixel-by-pixel will be flattened into memory-aligned cache arrays accelerated by compiler SIMD loop unrolling vectors and multi-threaded processing flags (`-O3`, OpenMP).

---

## 👥 Module Profile & AI Attributions
* **Module Owner**: Ssempewo Evalister - CNN Feature Block Architecture Builder)
* **Reviewer**: Joan 
* **AI Tool Implementation**: Assisted by Claude (Anthropic AI) for foundational layout drafts.
*