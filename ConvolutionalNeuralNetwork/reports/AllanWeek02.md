Report week 02

Author: Mugarura Allan

Assigned Module: Max Pooling Layer Implementation

Work done:
       1. Created max_pooling.hpp header that defines the maximum pooling spatial downsampling class layout.
       2. Created the implementation source file max_pooling.cpp that executes the sliding patch window traversal mechanics.
       3. Configured the downsampling structural logic to accept and process Agatha's custom 4D Tensor object natively.
       4. Integrated a parallel storage array map max_indices to track the exact flat 1D coordinates of maximum elements.
       5. Aligned the tracking logic with the mandated row-major flattening formula to prepare data for backpropagation.
       6. Created the verification script test_pooling.cpp with test code containing boundary assertions to test if my functions work well and as expected.

AI usage:
I used ChatGPT to map the localized 4D patch windows precisely into Agatha's underlying structural vector array. It also automated the verification calculations for the downsampled output dimensions and tracking coordinates.

challenge
  Configuring the argmax index tracker to extract the exact flat index from Agatha's layout initially caused spatial calculation offsets. To resolve this, I carefully expanded the index derivation inside the tracking nested loops and validated them using custom mock tensors.
  
conclusion
Week 2 was fully successful. The max pooling layer compiles cleanly, matches layout dimensions perfectly, and accurately records structural element indices ready for downstream backpropagation routing passes.