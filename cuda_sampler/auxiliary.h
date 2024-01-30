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

__forceinline__ __device__ void getRect(const FLOAT* p, FLOAT max_radius, uint2& rect_min, uint2& rect_max, const int* grid, const FLOAT* grid_offset) {
	rect_min = {
		(uint)min(grid[0], max((int)0, (int)((p[0] - grid_offset[0] - max_radius) / BLOCK_SIZE))),
		(uint)min(grid[1], max((int)0, (int)((p[1] - grid_offset[1] - max_radius) / BLOCK_SIZE)))
	};
	rect_max = {
		(uint)min(grid[0], max((int)0, (int)ceil((p[0] - grid_offset[0] + max_radius) / BLOCK_SIZE))),
		(uint)min(grid[1], max((int)0, (int)ceil((p[1] - grid_offset[1] + max_radius) / BLOCK_SIZE)))
	};
}

__forceinline__ __device__ uint2 getTile(const FLOAT* p, const int* grid, const FLOAT* grid_offset) {
	return {
		(uint)min(grid[0], max((int)0, (int)((p[0] - grid_offset[0]) / BLOCK_SIZE))),
		(uint)min(grid[1], max((int)0, (int)((p[1] - grid_offset[1]) / BLOCK_SIZE)))
	};
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
