
#include "image_loader.hpp"

#include <iostream>
#include <vector>

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

    vector<Pixel>& pixels = image.getPixels();

    if (pixels.empty())
    {
        cerr << "Image contains no pixels." << endl;
        return 1;
    }

    cout << endl;
    cout << "First pixel:" << endl;
    cout << "R: " << static_cast<int>(pixels[0].r) << endl;
    cout << "G: " << static_cast<int>(pixels[0].g) << endl;
    cout << "B: " << static_cast<int>(pixels[0].b) << endl;

    cout << endl << "Image loader test passed!" << endl;

    return 0;
}
