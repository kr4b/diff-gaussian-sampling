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
#include "aggregate_neighbors.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_gaussians", &PreprocessCUDA);
  m.def("sample_gaussians", &SampleGaussiansCUDA);
  m.def("sample_gaussians_backward", &SampleGaussiansBackwardCUDA);
  m.def("sample_gaussians_derivative", &SampleGaussiansDerivativeCUDA);
  m.def("sample_gaussians_derivative_backward", &SampleGaussiansDerivativeBackwardCUDA);
  m.def("sample_gaussians_laplacian", &SampleGaussiansLaplacianCUDA);
  m.def("sample_gaussians_laplacian_backward", &SampleGaussiansLaplacianBackwardCUDA);
  m.def("aggregate_neighbors", &AggregateNeighborsCUDA);
  m.def("aggregate_neighbors_backward", &AggregateNeighborsBackwardCUDA);
}
