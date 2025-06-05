# ==============================================
# File: block_sparse_matmul.py
# ==============================================
import torch
import block_sparse_ext

def block_sparse_matmul(A: torch.Tensor,
                        B: torch.Tensor,
                        mask: torch.Tensor,
                        tile_size: int = 32) -> torch.Tensor:
    """
    Forward block-sparse matrix multiply: C = A x B.

    Args:
      A: [M, K], float32 on CUDA
      B: [K, N], float32 on CUDA
      mask: 1D int32 tensor of size (M//tile_size)*(N//tile_size)
      tile_size: block tile size
    Returns:
      C: [M, N]
    """
    if A.dtype == torch.float16:
        A = A.to(torch.float32)  # Convert inside the kernel if needed
    if B.dtype == torch.float16:
        B = B.to(torch.float32)  # Convert inside the kernel if needed

    if mask.sum() == 0:
        # No active blocks => entire output is 0
        return torch.zeros(A.size(0), B.size(1), dtype=A.dtype, device=A.device)
    return block_sparse_ext.block_sparse_matmul(A, B, mask, tile_size)

def block_sparse_matmul_backward(A: torch.Tensor,
                                 grad_out: torch.Tensor,
                                 B: torch.Tensor,
                                 mask: torch.Tensor,
                                 tile_size: int = 32):
    """
    Backward pass that computes:
      dB = A^T x grad_out
      dA = grad_out x B^T
    in a block-sparse way (two separate kernels).
    Returns (dA, dB).
    """
    if mask.sum() == 0:
        return (torch.zeros_like(A), torch.zeros_like(B))

    dA, dB = block_sparse_ext.block_sparse_matmul_backward(A, grad_out, B, mask, tile_size)
    return (dA, dB)
