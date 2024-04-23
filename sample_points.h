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

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
PreprocessCUDA(
    const torch::Tensor& means,
    const torch::Tensor& values,
    const torch::Tensor& covariances,
    const torch::Tensor& conics,
    const torch::Tensor& samples,
    const bool debug);

torch::Tensor SampleGaussiansCUDA(
    const torch::Tensor& means,
    const torch::Tensor& values,
    const torch::Tensor& conics,
    const torch::Tensor& samples,
    const int num_rendered,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& sample_binning_buffer,
    const torch::Tensor& ranges,
    const torch::Tensor& sample_ranges,
    const bool debug);

torch::Tensor SampleGaussiansDerivativeCUDA(
    const torch::Tensor& means,
    const torch::Tensor& values,
    const torch::Tensor& conics,
    const torch::Tensor& samples,
    const int num_rendered,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& sample_binning_buffer,
    const torch::Tensor& ranges,
    const torch::Tensor& sample_ranges,
    const bool debug);

torch::Tensor SampleGaussiansLaplacianCUDA(
    const torch::Tensor& means,
    const torch::Tensor& values,
    const torch::Tensor& conics,
    const torch::Tensor& samples,
    const int num_rendered,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& sample_binning_buffer,
    const torch::Tensor& ranges,
    const torch::Tensor& sample_ranges,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SampleGaussiansBackwardCUDA(
    const torch::Tensor& means,
    const torch::Tensor& values,
    const torch::Tensor& conics,
    const torch::Tensor& samples,
    const int num_rendered,
    const torch::Tensor& dL_dout_values,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& sample_binning_buffer,
    const torch::Tensor& ranges,
    const torch::Tensor& sample_ranges,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SampleGaussiansDerivativeBackwardCUDA(
    const torch::Tensor& means,
    const torch::Tensor& values,
    const torch::Tensor& conics,
    const torch::Tensor& samples,
    const int num_rendered,
    const torch::Tensor& dL_dout_values,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& sample_binning_buffer,
    const torch::Tensor& ranges,
    const torch::Tensor& sample_ranges,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SampleGaussiansLaplacianBackwardCUDA(
    const torch::Tensor& means,
    const torch::Tensor& values,
    const torch::Tensor& conics,
    const torch::Tensor& samples,
    const int num_rendered,
    const torch::Tensor& dL_dout_values,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& sample_binning_buffer,
    const torch::Tensor& ranges,
    const torch::Tensor& sample_ranges,
    const bool debug);
