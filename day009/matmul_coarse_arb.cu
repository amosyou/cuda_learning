#define TILE_WIDTH 16
#define COARSE_FACTOR 4 // new
__global__ void matrixMulKernel(float* M, float* N, float* P, int Width) {

    // declare shared memory tiles for storing submatrices of M and N
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // compute block and thread indices
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // identify the row and column of the P matrix element being computed
    int Row = by * TILE_WIDTH + ty;
    // int Col = bx * TILE_WIDTH + tx;
    int ColStart = bx * TILE_WIDTH * COARSE_FACTOR + tx; // new

    // coarsening loop
    // float Pvalue = 0;
    float Pvalue[COARSE_FACTOR]; // new
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        Pvalue[c] = 0.0f;
    }

    // loop over tiles of M and N required to compute the output element
    // (phases in fig 5.8)
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {

        // collaborative loading a tile of M and a tile of N into shared memory
        // threads in the same block use the same shared memory
        // row index: Row
        // col index: ph*TILE_WIDTH + tx
        if ((Row < Width) && (ph*TILE_WIDTH + tx < Width)) {
            Mds[ty][tx] = M[Row*Width + (ph*TILE_WIDTH + tx)];
        } else {
            Mds[ty][tx] = 0.0f;
        }

        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int Col = ColStart + c * TILE_WIDTH; // new

            // row index: ph*TILE_WIDTH + ty
            // col index: Col
            if ((ph*TILE_WIDTH + ty < Width) && (Col < Width)) {
                Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
            } else {
                Nds[ty][tx] = 0.0f;
            }

            // synchronize to ensure all threads have loaded their values
            // (read-after-write dependence)
            __syncthreads();

            // compute the partial product for this tile
            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx]; // new
            }

            // synchronize again before loading the next tile
            // (write-after-read dependence)
            __syncthreads();
        }
    }

    // store the computed value into the output matrix P
    // NOTE: boundary check
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int Col = ColStart + c * TILE_WIDTH; // new

        if ((Row < Width) && (Col < Width)) {
            P[Row * Width + Col] = Pvalue[c]; // new
        }
    }
}