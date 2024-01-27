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

#ifndef CUDA_SAMPLER_H_INCLUDED
#define CUDA_SAMPLER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaSampler
{
    class Sampler
    {
    public:

        static int forward(
            std::function<char* (size_t)> geometryBuffer,
            std::function<char* (size_t)> binningBuffer,
            std::function<char* (size_t)> sample_binningBuffer,
            const int P, const int D, const int N, const int C,
            const int blocks,
            const int* tile_grid,
            const float* grid_offset,
            const float* means,
            const float* values,
            const float* covariances,
            const float* conics,
            const float* opacities,
            const float* samples,
            uint2* ranges,
            uint2* sample_ranges,
            float* out_values,
            float* radii = nullptr,
            bool debug = false);

        static void backward(
            const int P, const int D, const int N, const int C,
            const float* means,
            const float* values,
            const float* covariances,
            const float* conics,
            const float* opacities,
            const float* samples,
            const float* radii,
            const char* geomBuffer,
            const char* binningBuffer,
            const char* sample_binningBuffer,
            const uint2* ranges,
            const uint2* sample_ranges,
            const float* dL,
            float* dL_dmean,
            float* dL_dvalues,
            float* dL_dcovariances,
            float* dL_dconics,
            float* dL_dopacity,
            float* dL_dsamples,
            bool debug);
    };
};

#endif
