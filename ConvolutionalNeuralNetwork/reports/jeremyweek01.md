Atugonza Jeremy.
Personal Progressive Weekly Report.

Week 01 Report.

Task assigned: Image Representation/Data Structure.

Work done:
1. Modified the image.h header to define the Pixel structure and Image class for representing image data.
2. Created the Pixel structure to store the Red, Green and Blue (RGB) values of each pixel.
3. Used vector<Pixel> to store all the pixels of an image in memory.
4. Implemented the Image class in image.cpp, including width, height and channel information.
5. Implemented the getPixel() function to access individual pixels using their x and y coordinates.
6. Added coordinate bounds checking to prevent accessing pixels outside the image dimensions.
7. Added the setPixel() function to allow individual pixel values to be modified.
8. Connected the image representation structure with the existing image loading component so that loaded JPEG images are converted into Pixel objects.
9. Created and updated test_image.cpp to test the image representation and pixel access functions.
10. Successfully tested the implementation using the actual JPEG test image and verified the RGB values of individual pixels.

AI usage:
I used ChatGPT to understand how image data can be represented using structures and vectors in C++, and to understand how functions such as getPixel() and setPixel() can be implemented and used in the image representation.

Challenges:
The main challenge was understanding how a two-dimensional image can be stored in a one-dimensional vector<Pixel> and how the x and y coordinates are converted into the correct position in the vector. This was resolved by using the indexing formula y * width + x and testing it with an actual image.

Conclusion:
The image representation/data structure component was successfully implemented and tested. The system can now represent an image using RGB pixel data, access and modify individual pixels, and work together with the image loading component.
