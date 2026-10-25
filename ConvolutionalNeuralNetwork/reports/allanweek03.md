Report week 03
Author: Mugarura Allan
Assigned Module: Cross-Entropy Loss Function Module

Work done:
1. Created cross_entropy.hpp header that defines the categorical cross-entropy loss function class structure.
2. Created the implementation source file cross_entropy.cpp that executes the categorical error calculations.
3. Configured the error tracking loop to accept and process Agatha's custom 4D Tensor class shape formats natively.
4. Implemented a private numerical epsilon parameter to securely clip incoming probability boundary values.
5. Aligned the batch loss accumulation loops to match the project's mandated row-major layout dimensions.
6. Created the verification script test_cross_entropy.cpp with test code containing extreme boundary assertions to test if my functions work well and as expected.

AI usage:
I used ChatGPT to align the batch error aggregation loops precisely with Agatha's custom tensor dimensional indexing methods. It also automated the verification calculations for stable mathematical boundaries.

challenge
Configuring the error accumulator to handle extreme probability inputs of exactly 0.0 or 1.0 initially caused runtime math crashes. To resolve this, I implemented an element-wise thresholding bracket using a low epsilon buffer to clamp inputs before logarithmic processing.

conclusion
Week 3 was fully successful. The cross-entropy loss module compiles cleanly, matches layout dimensions perfectly, and accurately monitors loss bounds ready for downstream verification and optimization passes.