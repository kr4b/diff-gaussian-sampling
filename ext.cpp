/*
 * This is a modified version of the diff-gaussian-rasterization.
 * The original license still applies, see the original copyright notice below:
 *
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "sample_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_gaussians", &SampleGaussiansCUDA);
  m.def("sample_gaussians_backward", &SampleGaussiansBackwardCUDA);
  // m.def("sample_gaussians_derivatives", &SampleGaussiansCUDA);
  // m.def("sample_gaussians_derivatives_backward", &SampleGaussiansBackwardCUDA);
  // m.def("sample_gaussians_derivatives2", &SampleGaussiansCUDA);
  // m.def("sample_gaussians_derivatives2_backward", &SampleGaussiansBackwardCUDA);
}
