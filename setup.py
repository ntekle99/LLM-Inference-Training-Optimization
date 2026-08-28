from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_attn",
    version="0.1.0",
    packages=["custom_attn"],
    ext_modules=[
        CUDAExtension(
            name="custom_attn_C",
            sources=["csrc/bindings.cpp", "csrc/decode_attention.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
