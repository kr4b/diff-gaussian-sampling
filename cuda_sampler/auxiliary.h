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

#ifndef CUDA_SAMPLER_AUXILIARY_H_INCLUDED
#define CUDA_SAMPLER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

__forceinline__ __device__ void getRect(int D, const FLOAT* p, FLOAT max_radius, uint* rect_min, uint* rect_max, const int* grid, const FLOAT* grid_offset) {
	for (size_t i = 0; i < D; i++) {
		rect_min[i] = (uint)min(grid[i], max((int)0, (int)((p[i] - grid_offset[i] - max_radius) / BLOCK_SIZE)));
		rect_max[i] = (uint)min(grid[i], max((int)0, (int)ceil((p[i] - grid_offset[i] + max_radius) / BLOCK_SIZE)));
	}
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
