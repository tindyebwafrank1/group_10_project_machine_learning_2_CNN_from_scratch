# Week 3 Progress Report

## Completed
- Implemented `argmax` and batch `predict` in `include/prediction.hpp`
  and `src/prediction.cpp`.
- Wrote `tests/test_prediction.cpp` (normal cases, ties, negative scores,
  single class, empty input, NaN, batch of two samples).

## Prediction and Decision Logic
The network's final layer gives one probability (or score) per class. The
predicted class is the index of the largest value:

    predicted_class = argmax_k  p_k

- Ties go to the lowest index, so results are deterministic.
- Softmax is monotonic, so argmax of the logits equals argmax of the softmax
  output. Prediction works on either.
- For a batch, each sample is a block of K consecutive scores in the tensor
  data, and `predict` returns one label per sample.
- Empty input and NaN scores throw `std::invalid_argument`, because a NaN
  would silently give a wrong class.

## AI Use
- Tool: Claude
- Purpose: [generated a first draft of the header, implementation and tests.]
