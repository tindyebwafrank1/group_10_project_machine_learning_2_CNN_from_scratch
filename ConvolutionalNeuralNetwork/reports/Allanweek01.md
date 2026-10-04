Report week 01
Author: Mugarura Allan 

Assigned Module: Kernel and Filter Initialization Engine  
 

Work done:
         1. Created filter_initializer.hpp header that defines the kernel and filter initialization engine class.
         2. Created the implementation source file filter_initializer.cpp that implements the allocation and initialization processes declared in the header.
         3. Aligned the initialization data arrays natively with Agatha's custom Tensor object structural parameters to ensure compatibility.
         4. Integrated the deterministic std::mt19937 Mersenne Twister engine with a fixed testing seed value of 42 to populate the arrays uniformly.
         5. Implemented a dedicated bias tensor optimization layout structured at {1, K, 1, 1} to support downstream broadcasting equations.
         6. Created the verification script test_mugarura.cpp with test code containing boundary assertions to test if my functions work well and as expected.

AI usage:
    I used chartGPT to ensure perfect syntax integration with Agatha’s custom Tensor layout. It also automated the verification of our project's mandated 4D row-major flattening calculations.
    

    challenge
           Mapping 4D filters into Agatha's custom flat Tensor layout initially caused pointer mismatches. To resolve this, I strictly enforced the mandated row-major flattening formula and added boundary verification assertions.

conclusion
        Week 1 was fully successful. The initialization engine compiles cleanly, matches  layout perfectly, and delivers safe, deterministic weights ready for downstream convolution passes