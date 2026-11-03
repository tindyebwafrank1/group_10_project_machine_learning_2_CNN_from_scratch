

#include <iostream>
#include "image_loader.hpp"

using namespace std;

int main()
{
    Image image;

    // Load the image using a path relative to the project directory.
    if (!loadImage("data/cat01.jpg", image))
    {
        cerr << "Failed to load image: data/cat01.jpg" << endl;
        return 1;
    }

    cout << "Image loaded successfully!" << endl;
    cout << "Width: " << image.getWidth() << endl;
    cout << "Height: " << image.getHeight() << endl;
    cout << "Channels: " << image.getChannels() << endl;

    Pixel& originalPixel = image.getPixel(0, 0);

    cout << "Original Pixel (0,0): ";
    cout << "R=" << static_cast<int>(originalPixel.r);
    cout << " G=" << static_cast<int>(originalPixel.g);
    cout << " B=" << static_cast<int>(originalPixel.b) << endl;

    Pixel newPixel;
    newPixel.r = 255;
    newPixel.g = 0;
    newPixel.b = 0;

    image.setPixel(0, 0, newPixel);

    Pixel& changedPixel = image.getPixel(0, 0);

    cout << "Changed Pixel (0,0): ";
    cout << "R=" << static_cast<int>(changedPixel.r);
    cout << " G=" << static_cast<int>(changedPixel.g);
    cout << " B=" << static_cast<int>(changedPixel.b) << endl;

    return 0;
}
