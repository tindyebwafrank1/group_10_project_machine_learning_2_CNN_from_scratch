# Week 2 Report: Peter (M2, Operations Lead)

Sprint: Padding, strides, activations and pooling. Deadline: Saturday 10 Oct 2026, 12:30 AM.
Reviewer of my work: Frank.
 I review: Jeremy.

## 1. My coding responsibility: stride implementation and sub-sampling loop

| Item | File |
|---|---|
| Stride, output-size equation, window traversal, sub-sampling | `include/strides.hpp` |
| Unit tests (23 hand-checked combinations, brute-force sweep, traversal, convolution cross-check, sub-sampling, invalid input) | `tests/test_strides.cpp` |
| Demo | `examples/stride_demo.cpp` |
| Documentation (stride calculations and step traversal) | `docs/stride.md` |
| Build audit script | `scripts/check_build.sh` |

Public API: `Stride`, `isValidWindow`, `outputSize`, `WindowGrid`, `makeWindowGrid` (2 overloads),
`forEachWindow`, `paddedToInput`, `windowedOutputShape`, `subsampleSize`, `subsample`.

## 2. Acceptance criteria

| Criterion (Execution Guide) | Status | Evidence |
|---|---|---|
| Output shape equations validated across 15 distinct (H, W, P, S, K) combinations | Done | 23 combinations in `test_formula_combinations` plus exhaustive sweep |
| Strict build `g++ -O3 -Wall -Wextra -std=c++17` with no warnings | Run script, paste log | build log below |
| No memory leaks or out-of-bounds access | Run script, paste log | sanitizer / valgrind log below |
| Integration: used by convolution and pooling owners | Pending teammates | pending PRs from Joan, Mugarura, Evarista (see Dependencies) |

## 3. Environment:
 (Windows, g++ version)

## 4. Dependencies and hand-overs

| Who | What they need from me | Status |
|---|---|---|
| Joan (conv, avg pooling) | `makeWindowGrid`, `forEachWindow`, `paddedToInput` | message sent |
| Mugarura (max pooling) | grid for pooling windows, shared output formula | message sent |
| Evarista (FeatureBlock) | `windowedOutputShape`, `isValidWindow` | message sent |
| Frank (padding) | one shared output-size formula; `outputSize(in, K, P, S)` | message sent |
| Agatha | `Tensor` unchanged (public `data`, `shape`, `Tensor(n,c,h,w)`) | no change needed |

## 5. Supervision: Jeremy


- [ ] `relu(x) = max(0, x)`: negative to 0, zero stays 0, positive unchanged
- [ ] Shape preserved; input not modified unless an in-place function is clearly named
- [ ] Tests: negative, zero, positive, tiny and very large values, NaN behaviour decided and documented
- [ ] Stores what Week 4 needs (input or a 0/1 mask) so ReLU' can be computed
- [ ] Bad tensors (rank, zero dimension, data/shape mismatch) rejected with `std::invalid_argument`
- [ ] Compiles with `-O3 -Wall -Wextra -std=c++17` without warnings; sanitizer run clean
- [ ] No `using namespace std;` at global scope in a header
- [ ] PR description clear; small focused commits

Review outcome: (approved / changes requested, with comments)

## 6. Risks and blockers

| Risk | Impact | Action |
|---|---|---|
| Week-1 convolution has no stride/padding parameter | Joan must refactor in Week 2 | Joan to adopt `WindowGrid` now |
| Pooling and conv compute positions separately | Forward/backward mismatch in Week 4 | Everyone uses `WindowGrid` only |
| Open decision: how to hold 2D data in a 4D Tensor | Affects Week 3 flatten and dense layer | raise with Frank and Agatha |


# Stride: calculations and step traversal (stride.hpp)

Owner: Peter (Week 2). Reviewer: Frank. Header: `include/strides.hpp`. Tests: `tests/test_stride.cpp`. Demo: `examples/stride_demo.cpp`.


# REPORT PART TWO

## 1. Why this file exists

Convolution, max pooling and average pooling all slide a window over an image. If each of them
computes window positions on its own, the forward and backward passes can disagree by one index
and training silently fails. `strideS.hpp` is the single place that decides where windows are.
Everyone uses it; nobody re-implements the formula.

## 2. The equation

For one dimension, with input length `in`, window (kernel) size `K`, zero padding `P` on each side
and stride `S`:

    out = floor((in - K + 2P) / S) + 1

It is applied separately to height and width. Valid only when `in >= 1`, `K >= 1`, `S >= 1`
and `K <= in + 2P`. The window positions it counts are exactly the `p` with `p*S + K <= in + 2P`
(this is verified by brute force in the tests).

| in | K | P | S | out |
|---:|--:|--:|--:|----:|
| 5 | 3 | 0 | 1 | 3 |
| 5 | 3 | 0 | 2 | 2 |
| 5 | 3 | 1 | 1 | 5 (same size) |
| 7 | 3 | 1 | 2 | 4 |
| 28 | 5 | 0 | 1 | 24 |
| 28 | 5 | 2 | 1 | 28 (same size) |
| 224 | 7 | 3 | 2 | 112 |

## 3. Coordinates

* **Padded coordinates**: the image after padding, size `in + 2P`. Window `(oh, ow)` has its top-left
  corner at padded `(oh*S_h, ow*S_w)` and covers `K_h x K_w` cells.
* **Input coordinates**: padded coordinate minus `P`. A window near the border can start at a negative
  input coordinate (inside the padding).
* A padded cell that is inside the padding is an implicit zero. `paddedToInput()` returns `false` for
  those cells, so no padded copy of the image is needed.

## 4. API summary

| Function / type | What it does |
|---|---|
| `Stride(s)` / `Stride(sh, sw)` | Step sizes (default 1). |
| `isValidWindow(in, K, P, S, &reason)` | Non-throwing validity check with a reason. |
| `outputSize(in, K, P, S)` | The equation above; throws `std::invalid_argument` if invalid. |
| `makeWindowGrid(H, W, Kh, Kw, P, Stride)` | Builds a `WindowGrid` (also an overload with separate `pad_h`, `pad_w`). |
| `WindowGrid` | Holds sizes, `out_h`, `out_w`; `paddedRow/Col(o)` and `inputRow/Col(o)` give window corners. |
| `forEachWindow(grid, f)` | Visits windows row-major: `f(oh, ow, padded_row, padded_col)`. |
| `paddedToInput(grid, pr, pc, r, c)` | Maps a padded cell to an input cell; `false` if it is padding. |
| `windowedOutputShape({N,C,H,W}, Kh, Kw, P, Stride)` | Returns `{N, C, H_out, W_out}` (a convolution then replaces `C` with its filter count). |
| `subsampleSize(in, S, offset)` | `ceil((in - offset) / S)`. |
| `subsample(tensor, Stride, offset_h, offset_w)` | Keeps every S-th pixel: `out(n,c,i,j) = in(n,c, off_h + i*S_h, off_w + j*S_w)`. |

All functions throw `std::invalid_argument` with a message such as
`stride::outputSize: kernel is larger than the padded input`.

## 5. How other modules use it

**Convolution (forward and backward, Joan / Evarista)**

    const WindowGrid g = makeWindowGrid(H, W, Kh, Kw, pad, Stride(s));
    Tensor out(N, numFilters, g.out_h, g.out_w);
    forEachWindow(g, [&](size_t oh, size_t ow, size_t h0, size_t w0) {
        for (size_t n = 0; n < N; ++n)
            for (size_t k = 0; k < numFilters; ++k) {
                double sum = 0.0;
                for (size_t c = 0; c < C; ++c)
                    for (size_t r = 0; r < Kh; ++r)
                        for (size_t q = 0; q < Kw; ++q) {
                            size_t ir, ic;
                            if (paddedToInput(g, h0 + r, w0 + q, ir, ic))
                                sum += x(n, c, ir, ic) * kernel(k, c, r, q);
                        }
                out(n, k, oh, ow) = sum;
            }
    });

The backward pass loops over the same grid and uses the same `paddedToInput` test, so the index
mapping is identical by construction.

**Pooling (Mugarura, Joan)**: build the grid with `P = 0`, `Kh = Kw = window size`, and
`Stride(window)` (or the configured stride). Max pooling stores, per output cell, the input
coordinates of the maximum found while visiting that window; Week 4 routes gradients back to them.

**FeatureBlock (Evarista)**: call `windowedOutputShape` to size intermediate tensors and to validate
`(k_size, stride, pad)` early with `isValidWindow`.

## 6. Edge cases handled

| Case | Behaviour |
|---|---|
| `S = 0`, `K = 0`, `in = 0` | `std::invalid_argument` |
| `K > in + 2P` | `std::invalid_argument` (no window fits) |
| `K = in` or `K = in + 2P` | One window, `out = 1` |
| `in < K` but padding makes it fit (e.g. in=1, K=3, P=1) | `out = 1` |
| Stride larger than the window (e.g. in=2, K=2, S=5) | `out = 1`; cells between windows are skipped |
| Stride not dividing the length | Last partial step is dropped (floor) |
| Huge padding that would overflow `size_t` | `std::invalid_argument` |
| `subsample` offset `>= size`, or a corrupted tensor (bad rank, zero dim, data/shape mismatch) | `std::invalid_argument` |

## 7. Verification

* 23 hand-computed `(H, W, P, S, K)` combinations (acceptance criterion asks for 15), including
  non-square inputs, same-size padding and 224x224 / 28x28 layers.
* Exhaustive sweep (`in` 1..12, `P` 0..3, `S` 1..6, every valid `K`): equation equals a brute-force
  count of window positions; the last window fits and one more would not.
* Convolution through `WindowGrid` equals a convolution on an explicitly padded copy for
  `P` in 0..2 and `S` in 1..3.
* Sub-sampling: stride 1 = copy, uneven sizes, different strides per axis, offsets, multi-batch
  multi-channel, consistency with `outputSize(in, 1, 0, S)`.
* Run `./scripts/check_build.sh` for strict warnings, sanitizers and valgrind.

#####  ai USAGE
- used Open AI for to reference the structure of the strides program and the requirement for its development
