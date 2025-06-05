# ==============================================
# File: setup.py
# ==============================================
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='block_sparse_ext',
    version='0.0.1',
    author='Your Name',
    description='Block-Sparse MatMul Extension',
    ext_modules=[
        CUDAExtension(
            name='block_sparse_ext',
            sources=[
                'csrc/block_sparse_extension.cpp',
                'csrc/block_sparse_kernel.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
