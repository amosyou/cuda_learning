__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        int pixVal = 0;
        int numPixels = 0;

        // average of surrounding BLUR_SIZE x BLUR_SIZE
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; blurRow++) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; blurCol++) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // verify valid pixel in image
                if (curRow >= 0 && curRow >= 0 && curRow < h && curCol < w) {
                    pixVal += in[curRow * w + curCol];
                    ++pixels;
                }
            }
        }
        out[row * w + col] = (unsigned char) (pixVal / numPixels);
    }

}