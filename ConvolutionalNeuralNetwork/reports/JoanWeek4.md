# Week 4 Progress Rep

## Completed
- Implemented `convolution_backward` in `src/convolution_backward.cpp` with the interface
  in `include/convolution_backward.hpp`. It supports batches, multiple channels
  and filters, stride and zero padding.
- Added `conv_out_size` for the output size with padding and stride.
- Wrote `tests/test_convolution_backward.cpp`: a hand-computed case, finite-difference
  gradient checks on four configurations, and invalid-input tests.

## Backward Pass Mathematics
Forward: Y(n,k,i,j) = sum over c,r,s of Xpad(n,c,i*S+r,j*S+s) * K(k,c,r,s).
Each kernel weight and each input pixel affects the output only through the
windows they appear in, so by the chain rule:
- dK(k,c,r,s) = sum over n,i,j of dY(n,k,i,j) * Xpad(n,c,i*S+r,j*S+s)
- dX(n,c,h,w) = sum of dY(n,k,i,j) * K(k,c,r,s) over every window position
  (i,j) and kernel offset (r,s) with i*S+r-P = h and j*S+s-P = w.
The code loops over every output position once and adds each product to both
gradients. Positions in the padding are skipped.
Check by hand: with a 3x3 input 1..9, a 2x2 kernel of ones and dY of ones,
dK = [12 16; 24 28] and dX = [1 2 1; 2 4 2; 1 2 1] (the number of windows
covering each pixel).
Verification: the analytic gradients match central finite differences
(eps = 1e-6) to within 1e-5.

## Next Week
- Final report, final release checks, and the presentation.

## AI Use
- Tool: Claude
- Purpose: [Drafted the backward-pass code and tests.]
