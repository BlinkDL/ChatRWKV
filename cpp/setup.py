from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

use_openmp = True
print(f"  USE_OPEN: {use_openmp}")
use_avx512 = True
print(f"  USE_AVX512: {use_avx512}")

extra_compile_args = {"cxx": []}

if use_openmp:
    extra_compile_args["cxx"].append("-fopenmp")

if use_avx512:
    extra_compile_args["cxx"].append("-O3")
    extra_compile_args["cxx"].append("-march=skylake-avx512")
    extra_compile_args["cxx"].append("-mavx512f")
    extra_compile_args["cxx"].append("-DCPU_CAPABILITY_AVX512")
    extra_compile_args["cxx"].append("-Wno-array-bounds -Wno-unknown-pragmas")

setup(
    name='rwkv_cpp',
    ext_modules=[
        CppExtension(
            'rwkv_cpp',
            ['rwkv.cpp'],
            extra_compile_args=extra_compile_args),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
