EDRINE – GROUP 10

Personal Progressive Weekly Report

Week 04 Report

**Task Assigned:** Mini-batch Training Loop, Epoch Execution and Final Release Package.

**Work Done:**

1. Created `include/training.hpp` to define the training sample structure, training configuration, training history and softmax classifier interface.
2. Implemented `src/training.cpp` to calculate softmax probabilities and cross-entropy loss and perform gradient-based parameter updates.
3. Implemented mini-batch processing so that the training dataset is divided into smaller batches, including the final batch when it contains fewer samples than the configured batch size.
4. Implemented epoch execution to process all training mini-batches repeatedly for the specified number of epochs.
5. Recorded the average training loss after each epoch to monitor training progress.
6. Created `examples/week4_training_demo.cpp` to demonstrate model training and prediction using a small classification dataset.
7. Created `tests/test_training.cpp` to check epoch execution, loss reduction, mini-batch handling and predictions.
8. Prepared the Week 4 implementation and testing instructions for inclusion in the Group 10 final release package.

**Testing and Verification:**

- Compiled the Week 4 training demonstration using the C++17 compiler.
- Compiled and ran the automated training test.
- Checked whether training reduced the loss and whether the classifier predicted the expected labels.
- Recorded the actual test results for final verification.

**Challenges:**

1. Ensuring the training loop handles incomplete final mini-batches correctly.
2. Calculating stable softmax probabilities and cross-entropy loss.
3. Averaging gradients across each mini-batch before updating model parameters.
4. Separating the trainable classification head from the convolutional feature-extraction pipeline so the implementation's scope remains clear.

**Conclusion:**

Week 4 implements a mini-batch training loop, epoch execution, loss monitoring and gradient-based updates for a softmax classification head. The implementation and tests are intended to support the Group 10 final release. Final release status depends on successful compilation, execution of all project tests and integration verification with the complete CNN pipeline.
