// input image is encoded as unsigned chars [0, 255]
// each pixel is 3 consecutive chars for the 3 channels (RGB)
// L = 0.21 * r + 0.72 * g + 0.07 * b

__global__
void colorToGrayscaleConversion(unsigned char * Pout, unsigned char * Pin, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // left to right
    int row = blockIdx.y * blockDim.y + threadIdx.y; // up to down
    if (col < width && row < height) {

        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * 3; // can imagine rgb image having 3x as many columns than grayscale image

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[grayOffset] = 0.21f * r + 0.72f * g + 0.07f * b; // need f to denote float
    }
}