#include "image_loader.h"
#include <iostream>

using namespace std;

int main()
{
    cout<<endl;
    
    Image image;

    if (!loadImage("examples/cat01.jpg", image))
    {
        cout << "Failed to load image." <<endl;
        return 1;
    }

    else
    {
        cout << "Image loaded successfully!" <<endl;
    }

    cout << "Width: "<< image.getWidth()<<endl;
    cout << "Height: "<< image.getHeight()<<endl;
    cout << "Channels: "<< image.getChannels()<<endl;

    vector<Pixel>& pixels = image.getPixels();

    cout << endl;
    cout << "First pixel:" << endl;
    cout << "R: " << static_cast<int>(pixels[0].r) <<endl;
    cout << "G: " << static_cast<int>(pixels[0].g) <<endl;
    cout << "B: " << static_cast<int>(pixels[0].b) <<endl<<endl;

    return 0;
}