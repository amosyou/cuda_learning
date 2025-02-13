# CUDA Learning

cuda learning progression

resources:
- pmpp
- [cuda toolkit docs](https://docs.nvidia.com/cuda/)

---

## day 1

`vec_add.cu`

### summary

implemented a CUDA program for vector addition.

### learnings

- basics of writing CUDA kernel
    - `__global__` keyword
- two-level hierarchy
    - grid > blocks > threads
    - each block has < 1024 threads
- memory management of device (GPU) using `cudaMalloc`, `cudaMemcpy`, `cudaFree`
- compiling CUDA programs with nvcc

```
nvcc -o <program_name> <file_name>
```

### readings

pmpp ch1, ch2


## day 2

`color_to_grayscale.cu`

### summary

implemented CUDA kernel for converting images from rgb to grayscale

### learnings

- mapping threads to multi-dimensional layout
- row-major layout
    - `blockDim.x * gridDim.x` threads in horizontal direction
    - `col = blockIdx.x * blockDim.x + threadIdx.x`

### readings

pmpp ch3.1-3.2
