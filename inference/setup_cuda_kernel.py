import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

setup(
    name="cuda_kernel",
    ext_modules=[
        CUDAExtension(
            name="cuda_kernel",
            sources=[
                str(ROOT / "cuda_kernel.cpp"),
                str(ROOT / "cuda_kernel_impl.cu"),
            ],
            libraries=["cublas"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
