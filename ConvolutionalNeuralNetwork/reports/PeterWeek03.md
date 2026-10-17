# Week 3 Report

**Name:** Kweyamba Peter (Member 2, Operations Lead)
**Task:** Fully connected layer, forward pass `z = Wx + b`
**Reviewer:** Frank | **I review:** Jeremy (weight and bias initialization)

## 1. Objective

After convolution and pooling, the image is represented as feature maps. The fully connected layer converts these features into one score per class, where every output connects to every input [1]. My task was to implement the forward pass of this layer.

## 2. Deliverables

| File | Description |
|---|---|
| `include/fully_connected.hpp` | `FullyConnected` class (header only) |
| `tests/test_fully_connected.cpp` | Unit tests |
| `examples/fully_connected_demo.cpp` | Demonstration program |
| `scripts/check_build.sh` | Build and test script (updated) |

## 3. Mathematical Foundation

For each image `n` and output `o`:

    z[n][o] = b[o] + Σᵢ ( W[o][i] · x[n][i] )

where `W` is the weight matrix, `x` is the input, and `b` is the bias. This is a matrix multiplication followed by a bias addition [1][2].

**Example:** With `W = [[1, 2, 3], [4, 5, 6]]`, `b = [0.5, -1]`, and `x = [1, 0, -1]`:
- `z₀ = 1 + 0 − 3 + 0.5 = −1.5`
- `z₁ = 4 + 0 − 6 − 1 = −3`

This layer appears after the convolutional stages in LeNet-5 [2] and AlexNet [3]. Parameter count is `in × out + out` (e.g., 400 inputs to 120 outputs = 48,120 parameters). Computational cost per batch is approximately `N × in × out` multiply-adds.

Weights should not all be initialized identically. Xavier initialization [3] uses variance `2 / (fan_in + fan_out)`; He initialization uses `2 / fan_in` for ReLU networks. Jeremy's module generates these values, which my layer receives via `setWeights`. The backward pass in Week 4 requires the chain rule, so the layer caches its input.

## 4. Class Design

**Question 1: What class did I create?**
`class FullyConnected`

**Question 2: Why?**
The layer maintains state that must remain consistent: the weight vector must contain exactly `in × out` values, the bias vector exactly `out` values, and the cached input must be a true copy. A class enforces these invariants. My earlier work used free functions because no state needed preservation.

**Question 3: What data does it hold?**
- `in_features_`, `out_features_` — dimensions
- `weights_` — `vector<double>`, row-major {out, in}
- `bias_` — `vector<double>`
- `cached_input_` — saved copy of the last training input (for Week 4)

All members are private.

**Question 4: What functions operate on that data?**

| Function | Purpose | Access |
|---|---|---|
| `FullyConnected(in, out)` | Construct layer with zero-initialized parameters | public |
| `forward(input)` | Compute scores and cache input (training) | public |
| `infer(input)` | Compute scores without caching (inference) | public, `const` |
| `setWeights`, `setBias` | Replace parameters after validation | public |
| `weights()`, `bias()`, `weight(o, i)` | Read parameters | public, `const` |
| `inFeatures()`, `outFeatures()`, `parameterCount()` | Read dimensions | public, `const` |
| `hasCachedInput()`, `cachedInput()`, `clearCache()` | Manage cached input | public |
| `compute(input)` | Core `Wx + b` loop used by `forward` and `infer` | private |

**Question 5: Why are some members private and others public? How is it used?**
Private data prevents external code from resizing `weights_` (which would cause out-of-bounds reads) or inserting NaN values. Parameters can only be modified through `setWeights` and `setBias`, which validate sizes and values and leave the layer unchanged on failure. Getters are `const`, allowing inspection without modification. `compute` is private as a helper; `forward` and `infer` are public as the layer's interface.

In the pipeline: Frank's flatten step provides input; scores feed Agatha's softmax; Jeremy's initializer fills parameters; Edrine's pipeline calls `forward` for training and `infer` for testing; Week 4 uses the cached input and parameters.

**Additional design decisions:**
- Input accepts any 4D tensor where `C × H × W` equals the feature count, making the layer compatible with any flattening layout. Output shape is `{N, out, 1, 1}`.
- `forward` and `infer` are separate so inference cannot mutate the layer.
- `cached_input_` is `optional<Tensor>` because `Tensor` lacks a default constructor; `optional` clearly signals absence.

## 5. Limitations

The layer initializes to zero until an initializer fills it. Only the forward pass is implemented; backward pass and gradients follow in Week 4. The implementation uses a straightforward triple loop, so large layers will be slow.

## References

[1] I. Goodfellow, Y. Bengio and A. Courville, *Deep Learning*. Cambridge, MA, USA: MIT Press, 2016 (chapters 6 and 9, section 8.4).

[2] Y. LeCun, L. Bottou, Y. Bengio and H. Haffner, "Gradient-based learning applied to document recognition," *Proc. IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998.

[3] X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," in *Proc. 13th Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, PMLR 9, 2010, pp. 249–256.