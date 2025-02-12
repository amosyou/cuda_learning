// compute vector sum C = A + B

// kernel = function run on parallel threads
// global keyword indicates it's a CUDA C kernel
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// stub = host code for calling kernel
void vecAdd(float* A, float* B, float* C, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // part 1: allocate device memory for A, B, C
    // copy A and B to device memory
    cudaMalloc(&A_d, size);
    cudaMalloc(&B_d, size);
    cudaMalloc(&C_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // part 2: call kernel to launch kernel of threads
    // to perform the actual vector addition
    // first configuration param: # blocks in grid
    // second configuration param: # threads in block
    vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    // part 3: copy C from the device memory
    // free device vectors
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    const int N = 16;
    float A[N], B[N], C[N];

    vecAdd(A, B, C, N);
}