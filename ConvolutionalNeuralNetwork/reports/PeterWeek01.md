Image Preprocessing (preprocess.hpp), Week 1

All functions take and return `Tensor` with shape {N,C,H,W}, double precision, contiguous.
Index(n,c,h,w) = n*(C*H*W) + c*(H*W) + h*W + w. Errors throw std::invalid_argument.
Every function rejects: non-4-D tensors, zero dimensions, data/shape mismatch, NaN/Inf.

| Function | Input -> Output | Behaviour |
|---|---|---|
| fromInterleaved(ptr,count,H,W,C) | HWC buffer -> {1,C,H,W} | C is 1 or 3; no scaling; count must equal H*W*C |
| toInterleaved(t,n) | {N,C,H,W} -> HWC vector | for saving/inspecting |
| permuteChannels(t,order) / reverseChannels(t) | same shape | channel reorder, RGB<->BGR |
| rgbToGrayscale(t) | C=3 -> C=1 | 0.299R + 0.587G + 0.114B |
| grayscaleToRgb(t) | C=1 -> C=3 | replicates channel |
| toChannels(t,target) | C -> 1 or 3 | no-op if already target |
| resize(t,H',W',mode) | {N,C,H,W} -> {N,C,H',W'} | Nearest or Bilinear (half-pixel centres, clamped edges) |
| scaleToUnit(t,max) | same shape | x/max, max > 0 (255 for 8-bit) |
| standardize(t,mean,std) | same shape | (x-mean[c])/std[c], std > 0, one per channel |
| minMaxNormalize(t,lo,hi) | same shape | per sample into [lo,hi]; constant sample -> lo |
| computeChannelStats(t) | -> mean/std per channel | population std; constant channel -> std 1 |
| stackBatch(list) | -> {sum N,C,H,W} | all items must share C,H,W |
| preprocess(t,cfg) | -> tensor | reverse -> channels -> resize -> normalize |

Edge cases covered by tests: empty/mismatched buffers, wrong channel counts, invalid
permutations, zero/negative std, zero target sizes, constant images, non-square images,
NaN input, corrupted tensors.