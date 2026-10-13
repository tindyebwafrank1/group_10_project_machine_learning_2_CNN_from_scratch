# Feature Block Module (`cnn_library/feature_block.hpp`)

Owner: M7 Evalister Ssempewo | Reviewer: Joan  | Header-only, namespace `CNN`

## Purpose
Combines Conv2D -> Activation -> Pooling into one reusable block, and chains blocks
into a feature extractor that turns raw image tensors into condensed feature maps.

## Usage
```cpp
#include "cnn_library/feature_block.hpp"
using namespace CNN;

// Single block: in_c=1, out_c=8, 3x3 kernel, stride 1, pad 1
FeatureBlock block(1, 8, 3, 1, 1);
Tensor out = block.forward(image);            // image: (N, 1, H, W)

// Stack of blocks; input channels are inferred from the previous block
FeatureExtractor fx(1);
fx.add_block(4, 3, 1, 1).add_block(8, 3, 1, 1);
Tensor features = fx.forward(image);
auto shape = fx.output_shape(28, 28);         // {C, H, W}, no data needed
```

## API
| Member | Description |
|---|---|
| `FeatureBlock(in_c, out_c, k, stride, pad, act=ReLU, seed=42)` | builds the block, He-initialises filters |
| `block.conv`, `block.pool` | public members per the Week 2 interface; assign `block.pool = Pooling(3, 2, Pooling::Mode::Average)` to change pooling |
| `forward(input)` | conv -> activation -> pool; input is 4D (N, C, H, W) |
| `set_kernel(k)` | replace filters; shape must be {out_c, in_c, k, k} |
| `output_shape(H, W)` | `{out_c, H', W'}` without running data; throws if too small |
| `FeatureExtractor(in_c)` | builder; `add_block(...)` returns `*this` for chaining |
| `FeatureExtractor::forward / output_shape / num_blocks / block(i)` | run, inspect, access blocks |

`Activation` is one of `ReLU`, `Sigmoid`, `Tanh`.

## Formulas
- Conv: H_c = floor((H - K + 2P) / S) + 1 (same for W)
- Pool: H_p = floor((H_c - win) / pool_stride) + 1 (same for W)
- Kernel init: N(0, sqrt(2 / (in_c * K * K))), seeded with `seed` (builder uses base_seed + block index).

## Design decisions
- Default pooling is 2x2, stride 2, max (the Week 2 constructor has no pooling arguments).
- Shapes are validated before any computation, so errors say what is wrong instead of reading out of bounds.
- `forward` is `const`: no state is changed, so a block can be reused across batches.
- Seeds make results reproducible for tests and demos.

## Assumed interfaces (adapt if teammates' final APIs differ)
| From | Assumed |
|---|---|
| M4 `tensor.hpp` | `Tensor(n,c,h,w)`, `shape` (4 entries), `data`, `operator()` per Week 1 standard |
| `layers.hpp` | `Conv2D(stride, padding)`, `Tensor forward(const Tensor&, const Tensor& kernel) const`, **with padding implemented** (the Week 1 version ignored `padding`) |
| `activations.hpp` | `ReLU`, `Sigmoid`, `Tanh`, each with `Tensor forward(Tensor) const` |
| `pooling.hpp` | `Pooling(window, stride, Mode)`, `forward`, `window()`, `stride()` |

Only the four `#include`s and the `apply_activation()` / `forward()` bodies touch these.
`standins/` holds temporary versions for building before the real ones are merged.

## Limitations
- Square kernels and square pooling windows only.
- Pooling mode and size are set per block after construction, not in the constructor.
- Backward pass is Week 4 (pooling index tracking lives in the pooling module).
