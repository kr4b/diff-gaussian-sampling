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

#ifndef CUDA_SAMPLER_BACKWARD_H_INCLUDED
#define CUDA_SAMPLER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "config.h"
#include "sampler.h"

namespace BACKWARD {
    void render(
        const int D, const int C,
        const int blocks,
        const CudaSampler::Function function,
        const uint2* ranges,
        const uint2* sample_ranges,
        const uint32_t* point_list,
        const uint32_t* sample_point_list,
        const FLOAT* means,
        const FLOAT* values,
        const FLOAT* conics,
        const FLOAT* samples,
        const FLOAT* dL_dout_values,
        FLOAT* dL_dmeans,
        FLOAT* dL_dvalues,
        FLOAT* dL_dconics,
        FLOAT* dL_dsamples);
}

#endif
