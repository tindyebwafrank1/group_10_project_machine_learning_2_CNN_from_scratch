#include "image_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

using namespace std;

bool loadImage(const string& filename, Image& image)
{
    int width;
    int height;
    int channels;

    unsigned char* data = stbi_load(
        filename.c_str(),
        &width,
        &height,
        &channels,
        3
    );

    if (data == nullptr)
    {
        cout << "Error: Could not load image." << endl;
        return false;
    }

    image = Image(width, height, 3);

    vector<Pixel>& pixels = image.getPixels();

    pixels.resize(width * height);

    for (int i = 0; i < width * height; i++)
    {
        pixels[i].r = data[i * 3];
        pixels[i].g = data[i * 3 + 1];
        pixels[i].b = data[i * 3 + 2];
    }

    stbi_image_free(data);

    return true;
}