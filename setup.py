#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_sampling",
    packages=['diff_gaussian_sampling'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_sampling._C",
            sources=[
            "cuda_sampler/sampler_impl.cu",
            "cuda_sampler/forward.cu",
            # "cuda_sampler/backward.cu",
            "sample_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-g", "-Xcompiler", "-fno-gnu-unique", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
