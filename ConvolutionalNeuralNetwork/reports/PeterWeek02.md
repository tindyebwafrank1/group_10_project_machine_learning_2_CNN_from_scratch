# Week 2 Report: Kweyamba Peter
 **Role for week 2**: Operations and Risk Lead

**Sprint:**
 Padding, strides, activations and pooling.

*My reviewer*: Frank. I supervise: Jeremy.

## 1. My assignment

Stride implementation and the sub-sampling loop. Convolution, max pooling and average pooling all slide a
window over an image. If each of them worked out the window positions separately, the forward and backward
passes could disagree by one index and training would silently fail, so I wrote one shared place for the
window geometry.

The key equation, applied separately to height and width:

    out = floor((in - K + 2P) / S) + 1

with input length `in`, window size `K`, padding `P` on each side and stride `S`.

## 2. Class structure

### 2.1 Overview

| Type | Kind | Data members | Member functions | Access |
|---|---|---|---|---|
| `Stride` | struct | `h`, `w` (both `size_t`, default 1) | three constructors: default, `explicit Stride(both)`, `Stride(sh, sw)` | all public |
| `WindowGrid` | struct | `in_h`, `in_w`, `kernel_h`, `kernel_w`, `pad_h`, `pad_w`, `stride`, `out_h`, `out_w` | `paddedRow(oh)`, `paddedCol(ow)`, `inputRow(oh)`, `inputCol(ow)` (all `const`) | all public |
| `Tensor` (Agatha's, used not owned) | class | `data`, `shape` | constructor, `operator()` | `data` and `shape` public |

Free functions in `namespace cnn`: `isValidWindow`, `outputSize`, `makeWindowGrid` (two overloads),
`forEachWindow` (template), `paddedToInput`, `windowedOutputShape`, `subsampleSize`, `subsample`.
Internal helpers are in `namespace stride_detail` (`fail`, `checkWindow`, `requireValid`).

### 2.2 `Stride`

1. **What class did you create?** 
`struct Stride`.
2. **Why?**
 The step can differ for height and width. One small type keeps the two numbers together and
   makes calls readable (`Stride(2)` or `Stride(2, 4)`).
   
3. **What data does it hold?**
 `h` (row step) and `w` (column step), both default 1.

4. **What functions operate on that data?**
 Its three constructors: the default makes 1 by 1, the
   single-value constructor sets both, the two-value constructor sets each. The single-value one is
   `explicit`, so a plain integer is never converted to a `Stride` by accident.

5. **Private or public, and how is it used?** 
Public, because it is just two plain values with no rule to
   protect. A stride of 0 is rejected where it is used (`outputSize`, `makeWindowGrid`, `subsample`), which
   throw `std::invalid_argument`. Callers pass it to `makeWindowGrid` and `subsample`.

### 2.3 `WindowGrid`

1. **What class did you create?**
 `struct WindowGrid`.
2. **Why?**
     It is the single description of where sliding windows sit for one image plane. Convolution,
   pooling and the backward passes read the same grid, so their indices cannot drift apart.
3. **What data does it hold?** Input size (`in_h`, `in_w`), window size (`kernel_h`, `kernel_w`), padding on
   each side (`pad_h`, `pad_w`), the `stride`, and the number of window positions (`out_h`, `out_w`).
4. **What functions operate on that data?** 
    Member functions (all `const`): `paddedRow(oh)` and
   `paddedCol(ow)` give a window's top-left corner in padded coordinates; `inputRow(oh)` and `inputCol(ow)` give
   it in input coordinates (negative when the window starts inside the padding). Free functions:
   `makeWindowGrid` builds and validates the grid; `forEachWindow(grid, f)` visits every window in
   row-major order; `paddedToInput` maps a padded cell to an input cell and returns false when the cell is
   padding (an implicit zero), so no padded copy of the image is needed.
5. **Private or public, and how is it used?**
 All data members are public. The grid is meant to be created
   once by `makeWindowGrid` and then only read, usually
   as `const WindowGrid&` inside the inner loops. Public fields keep that code short and fast. The
   trade-off is that nothing stops code from changing a field after creation, so the grid's consistency
   depends on callers treating it as read-only. Making the fields private with read-only accessors
   would remove that risk.

### 2.4 Free functions and what is hidden

| Function | Purpose |
|---|---|
| `isValidWindow(in, K, P, S, &reason)` | non-throwing validity check with a reason |
| `outputSize(in, K, P, S)` | the equation; throws on invalid input |
| `makeWindowGrid(...)` | builds a `WindowGrid` (same padding on all sides, or separate height and width padding) |
| `forEachWindow(grid, f)` | traversal; a template so any callable (for example a lambda) can be passed |
| `paddedToInput(...)` | padded coordinate to input coordinate |
| `windowedOutputShape(shape, Kh, Kw, P, Stride)` | returns {N, C, H_out, W_out} |
| `subsampleSize`, `subsample` | keep every S-th pixel, with optional offsets |



### 2.5 How the classes are used by the program

* `subsample` reads and writes  `Tensor` through its public `data` and `shape` and builds the
  output with `Tensor(n, c, oh, ow)`.
* Planned use by team; convolution and average pooling
  on `WindowGrid`; And max pooling on it and stores the position of each maximum; 
  `FeatureBlock` uses `windowedOutputShape` and `isValidWindow` to check `(k_size, stride, pad)` early;
  padding code calls `outputSize` so there is only one copy of the formula.


## 3. Files and public interface

| File | Contents |
|---|---|
| `include/stride.hpp` | whole module (header-only) |
| `tests/test_stride.cpp` | unit tests |
| `examples/stride_demo.cpp` | demo of sizes, traversal, strided convolution, sub-sampling |
| `scripts/check_build.sh` | strict build, sanitizers and valgrind audit |
