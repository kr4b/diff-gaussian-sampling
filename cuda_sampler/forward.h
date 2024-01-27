/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(const int P, const int D, const int N, const int C,
		const float* means,
		const float* values,
		const float* covariances,
		const float* conics,
		const float* opacities,
		float* radii,
		const int* grid,
		const float* grid_offset,
		uint32_t* tiles_touched);

	// Main rasterization method.
	void render(
		const int D, const int C,
		const int blocks,
		const uint2* ranges,
		const uint2* sample_ranges,
		const uint32_t* point_list,
		const uint32_t* sample_point_list,
		const float* mean,
		const float* value,
		const float* conic,
		const float* opacity,
		const float* samples,
		float* out_value);
}


#endif
