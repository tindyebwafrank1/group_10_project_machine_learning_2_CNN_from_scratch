#include <iostream>
#include "image_loader.h"

using namespace std;

int main()
{
    Image image;

    if (!loadImage("examples/cat01.jpg", image))
    {
        cout << "Failed to load image." << endl;
        return 1;
    }

    cout << "Image loaded successfully!" << endl;
    cout << "Width: " << image.getWidth() << endl;
    cout << "Height: " << image.getHeight() << endl;
    cout << "Channels: " << image.getChannels() << endl;

    Pixel& originalPixel = image.getPixel(0, 0);

    cout << "Original Pixel (0,0): ";
    cout << "R=" << (int)originalPixel.r;
    cout << " G=" << (int)originalPixel.g;
    cout << " B=" << (int)originalPixel.b << endl;

    Pixel newPixel;
    newPixel.r = 255;
    newPixel.g = 0;
    newPixel.b = 0;

    image.setPixel(0, 0, newPixel);

    Pixel& changedPixel = image.getPixel(0, 0);

    cout << "Changed Pixel (0,0): ";
    cout << "R=" << (int)changedPixel.r;
    cout << " G=" << (int)changedPixel.g;
    cout << " B=" << (int)changedPixel.b << endl;

    return 0;
}