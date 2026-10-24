# Week 3 Progress Report

## Completed
- Implemented `ConfusionMatrix` in `include/cnn_library/metrics.hpp` (header-only).
- Accuracy, per-class and macro Precision, Recall, F1.
- Input validation (size mismatch, out-of-range labels, zero classes).
- Unit tests in `tests/test_metrics.cpp` (counts, metrics, perfect/worst cases, zero-division, invalid input, reset); clean under `-Wall -Wextra` and AddressSanitizer/UBSan.
- Example in `examples/metrics_example.cpp`; module doc in `docs/metrics.md`.

## In Progress
- PR to main for review by Joan .
- Agreeing the label interface with Joan's argmax output and Edrine's final app.

## Challenges/Blockers
- Delayed submission


## Next Week
- Support Week 4 integration: use metrics for per-epoch accuracy logs.
- Implement SGD optimizer step (W <- W - eta * grad W).

## AI Use
- Tool: Claude 
- Purpose: Generated the first draft of the metrics header, tests, example
- Reason: I reviewed and ran the code 
