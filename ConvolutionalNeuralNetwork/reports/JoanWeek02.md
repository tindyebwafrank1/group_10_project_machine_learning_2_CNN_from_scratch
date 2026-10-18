# Week 1 Progress Report

**Module:** Basic 2D convolution sliding window math

## Completed
- Declared the convolution interface in `include/cnn/convolution.hpp`:
  `conv_output_dim`, `window_dot` and `conv2d_single`.
- Implemented valid (no padding, stride 1) 2D convolution in
  `src/convolution.cpp` for a single-channel input and a single filter.
- Wrote `tests/test_convolution.cpp` covering the output size, an all-ones
  kernel, a diagonal kernel, and invalid input (kernel larger than input).
  Expected values were worked out by hand.
- Registered `convolution.cpp` and `test_convolution` in CMake.
- [Add: the PR link, and whether the tests passed on a clean build.]

## Convolution Mathematics
For a single channel, the output at position (h, w) is the dot product of the
kernel with the part of the input that sits under it:

    Y(h, w) = sum over r = 0..Kh-1, s = 0..Kw-1 of  X(h + r, w + s) * K(r, s)

The kernel slides across the input one step at a time (stride 1). At each
position, `window_dot` multiplies each kernel value by the input value under
it and adds the products. The kernel is not flipped, so this is
cross-correlation, which is what the project's formula specifies.

Without padding, the kernel must fit fully inside the input, so the output
size along each axis is:

    out = in - kernel + 1

Worked example: a 3x3 input with values 1 to 9 and a 2x2 kernel of ones gives
a 2x2 output. The first window is 1 + 2 + 4 + 5 = 12, and the other three are
16, 24 and 28.

For multiple channels and filters, the full formula adds a sum over input
channels c and an output index k.

## Next Week
- Rebase my early average pooling work onto `main` and align it with max pooling so the headers do not conflict.

## AI Use
- Tool Used: Claude
-I reviewed the code, checked the test values by hand, compiled it and edited it myself, and I can explain every function.