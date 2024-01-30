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

std::function<char*(size_t N)> resize_functional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
    auto float_opts = means.options().dtype(means.dtype());

    torch::Tensor radii = torch::full({P}, 0, float_opts);
    torch::Tensor out_values = torch::full({N, C}, 0.0, float_opts);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geom_buffer = torch::empty({0}, options.device(device));
    torch::Tensor binning_buffer = torch::empty({0}, options.device(device));
    torch::Tensor sample_binning_buffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geom_func = resize_functional(geom_buffer);
    std::function<char*(size_t)> binning_func = resize_functional(binning_buffer);
    std::function<char*(size_t)> sample_binning_func = resize_functional(sample_binning_buffer);

    const torch::Tensor min_bound = std::get<0>(samples.min(0));
    const torch::Tensor max_bound = std::get<0>(samples.max(0));

    const torch::Tensor tile_grid = torch::ceil((max_bound - min_bound + 1e-6f) / BLOCK_SIZE).to(torch::kInt32);
    const int blocks = torch::prod(tile_grid).item<int>();

    torch::Tensor ranges = torch::full({blocks * (long) sizeof(uint2) + 8}, 0, options.device(device));
    torch::Tensor sample_ranges = torch::full({blocks * (long) sizeof(uint2) + 8}, 0, options.device(device));

    int rendered = 0;
    if (P != 0) {
        rendered = CudaSampler::Sampler::preprocess(
          geom_func,
          binning_func,
          sample_binning_func,
          P, D, N, C, blocks,
          tile_grid.contiguous().data<int>(),
          min_bound.contiguous().data<FLOAT>(),
          means.contiguous().data<FLOAT>(),
          values.contiguous().data<FLOAT>(),
          covariances.contiguous().data<FLOAT>(),
          conics.contiguous().data<FLOAT>(),
          opacities.contiguous().data<FLOAT>(),
          samples.contiguous().data<FLOAT>(),
          reinterpret_cast<uint2*>(ranges.contiguous().data_ptr()),
          reinterpret_cast<uint2*>(sample_ranges.contiguous().data_ptr()),
          radii.contiguous().data<FLOAT>(),
          debug);
    }

    CudaSampler::Sampler::forward(
        P, D, N, C, blocks, rendered,
        means.contiguous().data<FLOAT>(),
        values.contiguous().data<FLOAT>(),
        conics.contiguous().data<FLOAT>(),
        opacities.contiguous().data<FLOAT>(),
        samples.contiguous().data<FLOAT>(),
        reinterpret_cast<char*>(binning_buffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(sample_binning_buffer.contiguous().data_ptr()),
        reinterpret_cast<uint2*>(ranges.contiguous().data_ptr()),
        reinterpret_cast<uint2*>(sample_ranges.contiguous().data_ptr()),
        radii.contiguous().data<FLOAT>(),
        out_values.contiguous().data<FLOAT>(),
        debug);

    return std::make_tuple(rendered, out_values, binning_buffer, sample_binning_buffer, ranges, sample_ranges);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SampleGaussiansBackwardCUDA(
    const torch::Tensor& means,
    const torch::Tensor& values,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& samples,
    const int num_rendered,
    const torch::Tensor& dL,
	const torch::Tensor& binning_buffer,
	const torch::Tensor& sample_binning_buffer,
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
    torch::Tensor dL_dconics = torch::zeros({P, D, D}, means.options());
    torch::Tensor dL_dopacities = torch::zeros({P}, means.options());
    torch::Tensor dL_dsamples = torch::zeros({N, D}, means.options());

    const torch::Tensor min_bound = std::get<0>(samples.min(0));
    const torch::Tensor max_bound = std::get<0>(samples.max(0));

    const torch::Tensor tile_grid = torch::ceil((max_bound - min_bound + 1e-6f) / BLOCK_SIZE).to(torch::kInt32);
    const int blocks = torch::prod(tile_grid).item<int>();

    if (P != 0) {
        CudaSampler::Sampler::backward(
            P, D, N, C, blocks, num_rendered,
            means.contiguous().data<FLOAT>(),
            values.contiguous().data<FLOAT>(),
            conics.contiguous().data<FLOAT>(),
            opacities.contiguous().data<FLOAT>(),
            samples.contiguous().data<FLOAT>(),
            reinterpret_cast<char*>(binning_buffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(sample_binning_buffer.contiguous().data_ptr()),
            reinterpret_cast<uint2*>(ranges.contiguous().data_ptr()),
            reinterpret_cast<uint2*>(sample_ranges.contiguous().data_ptr()),
            dL.contiguous().data<FLOAT>(),
            dL_dmeans.contiguous().data<FLOAT>(),
            dL_dvalues.contiguous().data<FLOAT>(),
            dL_dconics.contiguous().data<FLOAT>(),
            dL_dopacities.contiguous().data<FLOAT>(),
            dL_dsamples.contiguous().data<FLOAT>(),
            debug);
    }

    return std::make_tuple(dL_dmeans, dL_dvalues, dL_dconics, dL_dopacities, dL_dsamples);
}
