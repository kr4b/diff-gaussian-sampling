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

#include "config.h"

namespace CudaSampler {
    enum Function { gaussian, derivative, laplacian };

    class Sampler {

    public:

        static int preprocess(
            std::function<char* (size_t)> geometry_buffer,
            std::function<char* (size_t)> binning_buffer,
            std::function<char* (size_t)> sample_binning_buffer,
            const int P, const int D, const int N, const int C,
            const int blocks,
            const int* tile_grid,
            const FLOAT* grid_offset,
            const FLOAT* means,
            const FLOAT* values,
            const FLOAT* covariances,
            const FLOAT* conics,
            const FLOAT* opacities,
            const FLOAT* samples,
            uint2* ranges,
            uint2* sample_ranges,
            FLOAT* radii,
            bool debug = false);

        static void forward(
            const int P, const int D, const int N, const int C,
            const int blocks, const int num_rendered,
            const Function function,
            const FLOAT* means,
            const FLOAT* values,
            const FLOAT* conics,
            const FLOAT* opacities,
            const FLOAT* samples,
            char* binning_buffer,
            char* sample_binning_buffer,
            const uint2* ranges,
            const uint2* sample_ranges,
            FLOAT* out_values,
            bool debug = false);

        static void backward(
            const int P, const int D, const int N, const int C,
            const int blocks, const int num_rendered,
            const Function function,
            const FLOAT* means,
            const FLOAT* values,
            const FLOAT* conics,
            const FLOAT* opacities,
            const FLOAT* samples,
            char* binning_buffer,
            char* sample_binning_buffer,
            const uint2* ranges,
            const uint2* sample_ranges,
            const FLOAT* dL_dout_values,
            FLOAT* dL_dmeans,
            FLOAT* dL_dvalues,
            FLOAT* dL_dconics,
            FLOAT* dL_dopacities,
            FLOAT* dL_dsamples,
            bool debug);
    };
};

#endif
