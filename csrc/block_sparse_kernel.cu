#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include "block_sparse_kernel.h"

using namespace nvcuda;

#define TILE_SIZE 32
#define WARP_SIZE 32
#define BLOCK_SIZE_M 128
#define BLOCK_SIZE_N 128
#define BLOCK_SIZE_K 32
#define SHARED_MEM_PAD 8  // Avoid shared memory bank conflicts

__global__ void block_sparse_matmul_kernel(
    const half *A, const half *B, float *C,
    const int *mask, int M, int N, int K) {

    extern __shared__ half shared_mem[];
    half *tile_A = shared_mem;
    half *tile_B = shared_mem + (BLOCK_SIZE_M * BLOCK_SIZE_K) + SHARED_MEM_PAD;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int block_row = blockIdx.x * BLOCK_SIZE_M;
    int block_col = blockIdx.y * BLOCK_SIZE_N;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += BLOCK_SIZE_K) {
        if (mask[(block_row / TILE_SIZE) * (N / TILE_SIZE) + block_col / TILE_SIZE] == 0)
            continue;

        // Load tiles into shared memory manually
        int tile_offset_A = (block_row + warp_id * TILE_SIZE) * K + k;
        int tile_offset_B = k * N + block_col + lane_id * TILE_SIZE;

        tile_A[threadIdx.y * TILE_SIZE + threadIdx.x] = A[tile_offset_A];
        tile_B[threadIdx.y * TILE_SIZE + threadIdx.x] = B[tile_offset_B];
        __syncthreads();

        // Load into Tensor Core fragments
        wmma::load_matrix_sync(a_frag, tile_A, BLOCK_SIZE_K);
        wmma::load_matrix_sync(b_frag, tile_B, BLOCK_SIZE_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // Store results using atomicAdd to prevent race conditions
    int c_offset = (block_row + warp_id * TILE_SIZE) * N + block_col + lane_id * TILE_SIZE;
    atomicAdd(&C[c_offset], c_frag.x[0]);
}

extern "C" {
    void blockSparseMatMulLauncher(const half *A, const half *B, float *C, const int *mask, int M, int N, int K) {
        dim3 grid((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);
        dim3 block(WARP_SIZE * 8);  // 8 warps per block for better utilization
        size_t shared_mem_size = 2 * BLOCK_SIZE_M * BLOCK_SIZE_K * sizeof(half) + SHARED_MEM_PAD;

        block_sparse_matmul_kernel<<<grid, block, shared_mem_size>>>(A, B, C, mask, M, N, K);
        cudaDeviceSynchronize();
    }

    void blockSparseGradBLauncher(const half *A, const half *grad_out, float *dB, const int *mask, int M, int K, int N) {
        // Implement gradient B computation kernel launch here
    }

    void blockSparseGradALauncher(const half *grad_out, const half *B, float *dA, const int *mask, int M, int K, int N) {
        // Implement gradient A computation kernel launch here
    }
}
