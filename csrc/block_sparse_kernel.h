#ifndef BLOCK_SPARSE_KERNEL_H
#define BLOCK_SPARSE_KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

#define TILE_SIZE 32
#define WARP_SIZE 32
#define BLOCK_SIZE_M 128
#define BLOCK_SIZE_N 128
#define BLOCK_SIZE_K 32
#define SHARED_MEM_PAD 8

extern "C" {
    void launch_block_sparse_matmul(const half *A, const half *B, float *C, const int *mask, int M, int N, int K);
}

#endif // BLOCK_SPARSE_KERNEL_H
