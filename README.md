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
