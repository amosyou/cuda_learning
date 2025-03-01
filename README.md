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


## day 3

`image_blur.cu`, `matmul.cu`

### summary

implemented CUDA kernel for image blur and a basic matrix multiplication.

### learnings

- we verify that the row and col are both within the output range. otherwise, those threads should not execute.
- row and col thread indices should have 1:1 mapping with the output indices
- use multiple blocks to tile over the output

### readings

pmpp ch3.3-3.5


## day 4

### summary

learned about GPU architecture, sychronization, and scheduling

### learnings

- barrier synchronization similar to a checkpoint
- `__syncthreads() has to be executed by all threads in block. otherwise, deadlock/undefined behavior`
- warp = unit of thread scheduling in SMs
- blocks partitioned into warps for scheduling

### readings

pmpp ch4.1-4.4.5


## day 5

### summary

learned about control divergence, warp scheduling, occupancy

### learnings

- multiple passes for diff execution paths in control constructs (ie. if/else, for)
- latency tolerance/hiding = filling latency time of ops from one thread with others
- occupancy = ratio of # of warps assigned to SM / max number it supports

### readings

pmpp ch4.6-4.9


## day 6

`matmul_tiled.cu`

### summary

implemented a CUDA kernel for tiled matmul. learned about registers, shared memory, tiling.

### learnings

- loads from global memory make kernels very slow
- soln: use tiling to load small tiles into shared memory and run ops
- calculate indices using row/col indices + tile width + phase count

### readings

pmpp ch5.1-5.4
