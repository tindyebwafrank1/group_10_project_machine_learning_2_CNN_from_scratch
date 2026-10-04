#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <stdexcept>

using namespace std;

struct Pixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

class Image
{
private:
    int width;
    int height;
    int channels;

    vector<Pixel> pixels;

public:
    Image();

    Image(int width, int height, int channels);

    int getWidth() const;
    int getHeight() const;
    int getChannels() const;

    Pixel& getPixel(int x, int y);

    void setPixel(int x, int y, const Pixel& pixel);

    vector<Pixel>& getPixels();
    
};

#endif