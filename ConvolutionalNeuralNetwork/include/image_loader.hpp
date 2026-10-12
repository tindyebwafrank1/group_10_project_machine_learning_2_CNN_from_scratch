#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include "image.h"
#include <string>

using namespace std;

bool loadImage(const string& filename, Image& image);

#endif