#define TILE_WIDTH 16
__global__ void matrixMulKernel(float* M, float* N, float* P, int Width) {

    // declare shared memory tiles for storing submatrices of M and N
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // compute block and thread indices
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // identify the row and column of the P matrix element being computed
    // int Row = by * blockDim.y + ty;
    // int Col = bx * blockDim.x + tx;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // initialize an accumulator for the computed value
    float Pvalue = 0;

    // loop over tiles of M and N required to compute the output element
    // (phases in fig 5.8)
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {

        // collaborative loading a tile of M and a tile of N into shared memory
        // threads in the same block use the same shared memory
        // row index: Row
        // col index: ph*TILE_WIDTH + tx
        Mds[ty][tx] = M[Row*Width + (ph*TILE_WIDTH + tx)];
        // row index: ph*TILE_WIDTH + ty
        // col index: Col
        Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];

        // synchronize to ensure all threads have loaded their values
        // (read-after-write dependence)
        __syncthreads();

        // compute the partial product for this tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        // synchronize again before loading the next tile
        // (write-after-read dependence)
        __syncthreads();
    }

    // store the computed value into the output matrix P
    P[Row * Width + Col] = Pvalue;
}