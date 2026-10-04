#include "image.h"

using namespace std;

Image::Image()
{
    width = 0;
    height = 0;
    channels = 0;
}

Image::Image(int width, int height, int channels)
{
    this->width = width;
    this->height = height;
    this->channels = channels;

    pixels.resize(width * height);
}

int Image::getWidth() const
{
    return width;
}

int Image::getHeight() const
{
    return height;
}

int Image::getChannels() const
{
    return channels;
}

Pixel& Image::getPixel(int x, int y)
{
    if (x < 0 || x >= width || y < 0 || y >= height)
    {
        throw out_of_range("Pixel coordinates out of range");
    }

    return pixels[y * width + x];
}

void Image::setPixel(int x, int y, const Pixel& pixel)
{
    if (x < 0 || x >= width || y < 0 || y >= height)
    {
        throw out_of_range("Pixel coordinates out of range");
    }

    pixels[y * width + x] = pixel;
}

vector<Pixel>& Image::getPixels()
{
    return pixels;
}

