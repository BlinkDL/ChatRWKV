from setuptools import setup, Extension
from torch.utils import cpp_extension

# get current path
import os
current_path = os.path.dirname(os.path.abspath(__file__))

setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'quant_cuda', [f'{current_path}/quant_cuda.cpp', f'{current_path}/quant_cuda_kernel.cu']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)