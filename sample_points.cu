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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_sampler/config.h"
#include "cuda_sampler/sampler.h"
#include <fstream>
#include <string>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SampleGaussiansCUDA(
    const torch::Tensor& means,
    const torch::Tensor& values,
    const torch::Tensor& covariances,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& samples,
    const bool debug)
{
  const int P = means.size(0);
  const int D = means.size(-1);
  const int N = samples.size(0);
  const int C = values.size(-1);

  auto int_opts = means.options().dtype(torch::kInt32);
  auto float_opts = means.options().dtype(torch::kFloat32);

  torch::Tensor out_values = torch::full({N, C}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, float_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor sample_binningBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> sample_binningFunc = resizeFunctional(sample_binningBuffer);

  const torch::Tensor min_bound = std::get<0>(samples.min(0));
  const torch::Tensor max_bound = std::get<0>(samples.max(0));

  const torch::Tensor tile_grid = torch::ceil((max_bound - min_bound + 1e-6f) / BLOCK_SIZE).to(torch::kInt32);
  const int blocks = torch::prod(tile_grid).item<int>();

  torch::Tensor ranges = torch::full({blocks * (long) sizeof(uint2) + 8}, 0, options.device(device));
  torch::Tensor sample_ranges = torch::full({blocks * (long) sizeof(uint2) + 8}, 0, options.device(device));

  int rendered = 0;
  if(P != 0) {
      rendered = CudaSampler::Sampler::forward(
        geomFunc,
        binningFunc,
        sample_binningFunc,
        P, D, N, C, blocks,
        tile_grid.contiguous().data<int>(),
        min_bound.contiguous().data<float>(),
        means.contiguous().data<float>(),
        values.contiguous().data<float>(),
        covariances.contiguous().data<float>(),
        conics.contiguous().data<float>(),
        opacities.contiguous().data<float>(),
        samples.contiguous().data<float>(),
        reinterpret_cast<uint2*>(ranges.contiguous().data_ptr()),
        reinterpret_cast<uint2*>(sample_ranges.contiguous().data_ptr()),
        out_values.contiguous().data<float>(),
        radii.contiguous().data<float>(),
        debug);
  }

  return std::make_tuple(rendered, out_values, radii, geomBuffer, binningBuffer, sample_binningBuffer, ranges, sample_ranges);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SampleGaussiansBackwardCUDA(
    const torch::Tensor& means,
	const torch::Tensor& radii,
    const torch::Tensor& values,
    const torch::Tensor& conics,
    const torch::Tensor& covariances,
    const torch::Tensor& opacities,
    const torch::Tensor& samples,
    const int num_rendered,
    const torch::Tensor& dL,
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& sample_binningBuffer,
    const torch::Tensor& ranges,
	const torch::Tensor& sample_ranges,
    const bool debug)
{
  const int P = means.size(0);
  const int D = means.size(-1);
  const int N = samples.size(0);
  const int C = values.size(-1);
  
  torch::Tensor dL_dmeans = torch::zeros({P, D}, means.options());
  torch::Tensor dL_dvalues = torch::zeros({P, C}, means.options());
  torch::Tensor dL_dcovariances = torch::zeros({P, D, D}, means.options());
  torch::Tensor dL_dconics = torch::zeros({P, D, D}, means.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means.options());
  torch::Tensor dL_dsamples = torch::zeros({P, D}, means.options());
  
  if(P != 0) {  
      CudaSampler::Sampler::backward(P, D, N, C,
        means.contiguous().data<float>(),
        values.contiguous().data<float>(),
        covariances.contiguous().data<float>(),
        conics.contiguous().data<float>(),
        opacities.contiguous().data<float>(),
        samples.contiguous().data<float>(),
        radii.contiguous().data<float>(),
        reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(sample_binningBuffer.contiguous().data_ptr()),
        reinterpret_cast<uint2*>(ranges.contiguous().data_ptr()),
        reinterpret_cast<uint2*>(sample_ranges.contiguous().data_ptr()),
        dL.contiguous().data<float>(),
        dL_dmeans.contiguous().data<float>(),
        dL_dvalues.contiguous().data<float>(),
        dL_dcovariances.contiguous().data<float>(),
        dL_dconics.contiguous().data<float>(),
        dL_dopacity.contiguous().data<float>(),
        dL_dsamples.contiguous().data<float>(),
        debug);
  }

  return std::make_tuple(dL_dmeans, dL_dvalues, dL_dcovariances, dL_dconics, dL_dopacity, dL_dsamples);
}

