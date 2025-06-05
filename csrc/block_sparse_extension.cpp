// ==============================================
// File: block_sparse_extension.cpp
// ==============================================
#include <torch/extension.h>
#include <tuple>
#include <vector>

extern "C" {
// Forward
void blockSparseMatMulLauncher(
    const float* A,
    const float* B,
    float* C,
    const int* block_mask,
    int M, int K, int N,
    int tile_size
);

// dB = A^T x grad_out
void blockSparseGradBLauncher(
    const float* A,
    const float* grad_out,
    float* dB,
    const int* block_mask,
    int M, int K, int N,
    int tile_size
);

// dA = grad_out x B^T
void blockSparseGradALauncher(
    const float* grad_out,
    const float* B,
    float* dA,
    const int* block_mask,
    int M, int K, int N,
    int tile_size
);
} // extern "C"


// Forward: C = A * B
torch::Tensor block_sparse_matmul(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor block_mask,
    int tile_size
) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");
    TORCH_CHECK(block_mask.is_cuda(), "block_mask must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");

    int M = A.size(0);
    int K = A.size(1);
    TORCH_CHECK(B.size(0) == K, "B must have size(0)=K");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    blockSparseMatMulLauncher(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        block_mask.data_ptr<int>(),
        M, K, N,
        tile_size
    );

    return C;
}


// Backward: compute dB and dA in two kernels
//   dB = A^T x grad_out  => shape(K,N)
//   dA = grad_out x B^T => shape(M,K)
std::tuple<torch::Tensor, torch::Tensor> block_sparse_matmul_backward(
    torch::Tensor A,
    torch::Tensor grad_out,
    torch::Tensor B,
    torch::Tensor block_mask,
    int tile_size
) {
    TORCH_CHECK(A.is_cuda(),        "A must be CUDA");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(B.is_cuda(),        "B must be CUDA");
    TORCH_CHECK(block_mask.is_cuda(),"block_mask must be CUDA");

    TORCH_CHECK(A.dtype() == torch::kFloat32,        "A must be float32");
    TORCH_CHECK(grad_out.dtype() == torch::kFloat32, "grad_out must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32,        "B must be float32");

    int M = A.size(0);
    int K = A.size(1);
    TORCH_CHECK(grad_out.size(0) == M, "grad_out.size(0) must match A.size(0)");
    TORCH_CHECK(B.size(0) == K, "B.size(0) must match K");
    TORCH_CHECK(grad_out.size(1) == B.size(1), "grad_out.size(1) must match B.size(1)");
    int N = B.size(1);

    auto dB = torch::zeros({K, N}, B.options());
    auto dA = torch::zeros({M, K}, A.options());

    // 1) dB = A^T x grad_out
    blockSparseGradBLauncher(
        A.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        dB.data_ptr<float>(),
        block_mask.data_ptr<int>(),
        M, K, N,
        tile_size
    );

    // 2) dA = grad_out x B^T
    blockSparseGradALauncher(
        grad_out.data_ptr<float>(),
        B.data_ptr<float>(),
        dA.data_ptr<float>(),
        block_mask.data_ptr<int>(),
        M, K, N,
        tile_size
    );

    return std::make_tuple(dA, dB);
}


// PyBind module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_sparse_matmul",
          &block_sparse_matmul,
          "Block-Sparse MatMul (CUDA)");

    m.def("block_sparse_matmul_backward",
          &block_sparse_matmul_backward,
          "Block-Sparse MatMul Backward (CUDA)");
}
