#!/usr/bin/env python3
# ====================================================
# demo.py
# Compare Dense vs. Block-Sparse MatMul Performance
# ====================================================

import time
import torch
from block_sparse_matmul import block_sparse_matmul, block_sparse_matmul_backward

def measure_cuda_time(func, *args, runs=10):
    """
    Measures average CUDA time in seconds for a given function.
    """
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        outputs = func(*args)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / runs, outputs

def dense_forward_backward(A, B, grad_out):
    """
    Dense forward + backward:
      Forward:   C = A @ B
      Backward:  dA = grad_out @ B.t()
                 dB = A.t() @ grad_out
    Returns (C, (dA, dB)).
    """
    # Forward
    C = A @ B
    # Backward
    dA = grad_out @ B.t()
    dB = A.t() @ grad_out
    return C, (dA, dB)

def sparse_forward_backward(A, B, grad_out, mask, tile_size=32):
    """
    Block-sparse forward + backward using custom CUDA kernels:
      Forward:   C = block_sparse_matmul(A, B, mask)
      Backward:  (dA, dB) = block_sparse_matmul_backward(A, grad_out, B, mask)
    Returns (C, (dA, dB)).
    """
    # Forward
    C = block_sparse_matmul(A, B, mask, tile_size)
    # Backward
    dA, dB = block_sparse_matmul_backward(A, grad_out, B, mask, tile_size)
    return C, (dA, dB)

def create_block_mask(M, N, tile_size=32, keep_fraction=0.5, device='cuda'):
    """
    Create a random block mask for a matrix of shape [M, N].
    keep_fraction = fraction of blocks to keep as '1' (active).
    The mask is shape [ (M//tile_size) * (N//tile_size) ] with int32 dtype.
    """
    blocks_m = (M + tile_size - 1) // tile_size
    blocks_n = (N + tile_size - 1) // tile_size
    total_blocks = blocks_m * blocks_n

    # Randomly choose which blocks are active
    rand_vals = torch.rand(total_blocks, device=device)
    mask = (rand_vals < keep_fraction).int()  # e.g. 50% blocks active
    return mask

def compare_dense_vs_sparse(M, K, N, tile_size, sparsity=0.5, runs=10, device="cuda"):
    """
    Generate random A, B, grad_out, create a block mask with 'sparsity' fraction
    of blocks turned OFF, then measure dense vs. block-sparse times.
    """
    # Invert logic: keep_fraction = 1 - sparsity (i.e. fraction of blocks that remain 1)
    keep_fraction = 1.0 - sparsity

    # Create random data
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)
    grad_out = torch.randn(M, N, device=device, dtype=torch.float32)

    # Create block mask with some fraction of blocks = 1
    mask = create_block_mask(M, N, tile_size=tile_size, keep_fraction=keep_fraction, device=device)

    # Measure Dense
    dense_forward_time, (denseC, (dense_dA, dense_dB)) = measure_cuda_time(dense_forward_backward, A, B, grad_out, runs=runs)
    dense_backward_time = dense_forward_time  # Because we do forward+backward in one call

    # Measure Sparse
    sparse_forward_time, (sparseC, (sparse_dA, sparse_dB)) = measure_cuda_time(sparse_forward_backward, A, B, grad_out, mask, tile_size, runs=runs)
    sparse_backward_time = sparse_forward_time

    # Check correctness: sums or L2 difference
    diff_C = (sparseC - denseC).norm().item()
    diff_dA = (sparse_dA - dense_dA).norm().item()
    diff_dB = (sparse_dB - dense_dB).norm().item()

    # Print results
    print(f"\n=== Compare Dense vs. Sparse for shape A=[{M},{K}], B=[{K},{N}] ===")
    print(f"Tile size: {tile_size}, Active blocks fraction ~{keep_fraction:.2f}")

    print(f"Dense FWD+BWD: {dense_forward_time*1000:.3f} ms (avg of {runs} runs)")
    print(f"Sparse FWD+BWD: {sparse_forward_time*1000:.3f} ms (avg of {runs} runs)")
    if sparse_forward_time > 1e-9:
        speedup = dense_forward_time / sparse_forward_time
    else:
        speedup = 1.0

    print(f"Speedup (Dense / Sparse): {speedup:.2f}x\n")

    print("Check correctness (L2 differences):")
    print(f"  ||C_dense - C_sparse||   = {diff_C:.4f}")
    print(f"  ||dA_dense - dA_sparse|| = {diff_dA:.4f}")
    print(f"  ||dB_dense - dB_sparse|| = {diff_dB:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    # Example shapes to test
    test_cases = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ]
    tile_size = 32
    runs = 10

    for (M, K, N) in test_cases:
        # We'll set 50% sparsity => keep_fraction=0.5
        # Means half the blocks are turned off
        compare_dense_vs_sparse(M, K, N, tile_size=tile_size, sparsity=0.5, runs=runs, device=device)


if __name__ == "__main__":
    main()
