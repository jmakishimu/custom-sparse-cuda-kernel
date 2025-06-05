import torch
from block_sparse_matmul import block_sparse_matmul, block_sparse_matmul_backward
from time import perf_counter

def measure_cuda_time(func, *args, runs=10, **kwargs):
    """
    Measures the average CUDA execution time (in seconds) of a given function.
    """
    # Warm-up runs (to stabilize CUDA)
    for _ in range(2):
        _ = func(*args, **kwargs)

    torch.cuda.synchronize()
    start = perf_counter()

    for _ in range(runs):
        out = func(*args, **kwargs)
    torch.cuda.synchronize()

    end = perf_counter()
    avg_time = (end - start) / runs
    return avg_time, out

def main():
    # Example shapes
    M, K, N = 64, 64, 64
    tile_size = 32

    # Generate random inputs
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    grad_output = torch.rand(M, N).cuda()

    # Example block mask: 50% chance each block is active
    block_mask = (torch.rand((M // tile_size, N // tile_size)) > 0.5).cuda()

    # --- Forward pass timing ---
    forward_time, forward_out = measure_cuda_time(
        block_sparse_matmul,
        A, B, block_mask,
        runs=10,
        tile_size=tile_size
    )
    print(f"⏳ Forward Kernel Time: {forward_time:.6f} sec")

    # --- Backward pass timing ---
    # Here, we measure the backward kernel by providing the same A and grad_output
    backward_time, (dA, dB) = measure_cuda_time(
        block_sparse_matmul_backward,
        A, grad_output, block_mask,
        runs=10,
        tile_size=tile_size
    )
    print(f"⏳ Backward Kernel Time: {backward_time:.6f} sec")

    # Print ratio or difference
    ratio = backward_time / forward_time if forward_time > 0 else float('inf')
    print(f"Backward is {ratio:.2f}x slower than Forward")

    # Optionally, check the shape or any debug info
    print("Forward Output Shape:", forward_out.shape)
    print("dA Shape:", dA.shape, "| dB Shape:", dB.shape)

if __name__ == "__main__":
    main()
