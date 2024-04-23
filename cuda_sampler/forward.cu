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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define NUM_THREADS 256

// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(
    const int P, const int D, const int N, const int C,
    const FLOAT* means,
    const FLOAT* values,
    const FLOAT* covariances,
    const FLOAT* conics,
    FLOAT* radii,
    const int* grid,
    const FLOAT* grid_offset,
    uint32_t* tiles_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0.0;
	tiles_touched[idx] = 0;

    const FLOAT* cov = covariances + idx * D * D;
    FLOAT my_radius = 0.0;
    uint32_t touched = 0;

	// Compute extent in screen space (by finding eigenvalues of
	// the covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
    if (D == 1) {
        my_radius = 3.0 * sqrt(cov[0]);
    } else if (D == 2) {
        FLOAT det = (cov[0] * cov[3] - cov[1] * cov[1]);
        if (det == 0.0)
            return;
        FLOAT mid = 0.5f * (cov[0] + cov[3]);
        FLOAT lambda = mid + sqrt(max(1e-6, mid * mid - det));
        my_radius = 3.0 * sqrt(lambda);
    }

    uint* rect_min = new uint[D];
    uint* rect_max = new uint[D];

    getRect(D, means + idx * D, my_radius, rect_min, rect_max, grid, grid_offset);

    if (D == 1) {
        touched = rect_max[0] - rect_min[0];
    } else if (D == 2) {
        touched = (rect_max[1] - rect_min[1]) * (rect_max[0] - rect_min[0]);
    }

    delete rect_min;
    delete rect_max;

    if (touched == 0)
        return;

	// Store some useful helper data for the next steps.
	radii[idx] = my_radius;
	tiles_touched[idx] = touched;
}

typedef void(*gaussian_func)(const FLOAT*, const FLOAT*, const FLOAT*, FLOAT*, int, int, int);

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <gaussian_func F>
__global__ void renderCUDA(
    const int D, const int C,
	const uint2* __restrict__ ranges,
	const uint2* __restrict__ sample_ranges,
	const uint32_t* __restrict__ point_list,
	const uint32_t* __restrict__ sample_point_list,
	const FLOAT* __restrict__ means,
	const FLOAT* __restrict__ values,
	const FLOAT* __restrict__ conics,
    const FLOAT* __restrict__ samples,
	FLOAT* __restrict__ out_values)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();

	// Load start/end range of IDs to process in bit sorted list.
	const uint2 range = ranges[block.group_index().x];
	const int rounds = ((range.y - range.x + NUM_THREADS - 1) / NUM_THREADS);
	int toDo = range.y - range.x;

	const uint2 sample_range = sample_ranges[block.group_index().x];
    const int num_samples = sample_range.y - sample_range.x;
    const int samples_per_thread = (num_samples + NUM_THREADS - 1) / NUM_THREADS;
    const int sample_offset = block.thread_rank() * samples_per_thread;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[NUM_THREADS];

    FLOAT* X = new FLOAT[D];

	// Iterate over batches until range is complete
	for (int i = 0; i < rounds; i++, toDo -= NUM_THREADS) {
        block.sync();

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * NUM_THREADS + block.thread_rank();
		if (range.x + progress < range.y) {
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
		}
		block.sync();

        if (sample_offset >= num_samples)
            continue;

		// Iterate over current batch
		for (int j = 0; j < min(NUM_THREADS, toDo); j++) {
            const int id = collected_id[j];
            const FLOAT* mean = means + id * D;
            const FLOAT* con = conics + id * D * D;
            const FLOAT* value = values + id * C;

            for (int s = 0; s < samples_per_thread; s++) {
                if (sample_offset + s >= num_samples)
                    break;

                const int sample_id = sample_point_list[sample_range.x + sample_offset + s];
                const FLOAT* sample = samples + sample_id * D;

                for (int k = 0; k < D; k++)  X[k] = mean[k] - sample[k];

                F(X, con, value, out_values, sample_id, D, C);
            }
        }
    }

    delete X;
}

__forceinline__ __device__ void gaussian(const FLOAT* X, const FLOAT* con, const FLOAT* values, FLOAT* out_values, int sample_id, int D, int C) {
    if (D == 1) {
        const FLOAT power = -0.5 * con[0] * X[0] * X[0];
        if (power > 0.0) return;

        const FLOAT alpha = exp(power);
        for (int ch = 0; ch < C; ch++)
            out_values[sample_id * C + ch] += values[ch] * alpha;
    } else if (D == 2) {
        const FLOAT power = -0.5 * (con[0] * X[0] * X[0] + con[3] * X[1] * X[1]) - con[1] * X[0] * X[1];
        if (power > 0.0) return;

        const FLOAT alpha = exp(power);
        for (int ch = 0; ch < C; ch++)
            out_values[sample_id * C + ch] += values[ch] * alpha;
    }
}

__forceinline__ __device__ void gaussian_derivative(const FLOAT* X, const FLOAT* con, const FLOAT* values, FLOAT* out_values, int sample_id, int D, int C) {
    if (D == 1) {
        const FLOAT x1 = con[0] * X[0];
        const FLOAT power = -0.5 * x1 * X[0];
        if (power > 0.0) return;

        const FLOAT alpha = exp(power);
        for (int ch = 0; ch < C; ch++) {
            out_values[(sample_id * D + 0) * C + ch] += values[ch] * alpha * x1;
        }
    } else if (D == 2) {
        const FLOAT x1 = con[0] * X[0];
        const FLOAT x2 = con[3] * X[1];
        const FLOAT power = -0.5 * (x1 * X[0] + x2 * X[1]) - con[1] * X[0] * X[1];
        if (power > 0.0) return;

        const FLOAT alpha = exp(power);
        for (int ch = 0; ch < C; ch++) {
            out_values[(sample_id * D + 0) * C + ch] += values[ch] * alpha * (x1 + con[1] * X[1]);
            out_values[(sample_id * D + 1) * C + ch] += values[ch] * alpha * (x2 + con[1] * X[0]);
        }
    }
}

__forceinline__ __device__ void gaussian_laplacian(const FLOAT* X, const FLOAT* con, const FLOAT* values, FLOAT* out_values, int sample_id, int D, int C) {
    if (D == 1) {
        const FLOAT x1 = con[0] * X[0];
        const FLOAT power = -0.5 * x1 * X[0];
        if (power > 0.0) return;

        const FLOAT alpha = exp(power);
        for (int ch = 0; ch < C; ch++) {
            out_values[(sample_id * D + 0) * C + ch] += values[ch] * alpha * (x1 * x1 - con[0]);
        }
    } else if (D == 2) {
        const FLOAT x1 = con[0] * X[0];
        const FLOAT x2 = con[3] * X[1];
        const FLOAT power = -0.5 * (x1 * X[0] + x2 * X[1]) - con[1] * X[0] * X[1];
        if (power > 0.0) return;

        const FLOAT a1 = x1 + con[1] * X[1];
        const FLOAT a2 = x2 + con[1] * X[0];

        const FLOAT alpha = exp(power);
        for (int ch = 0; ch < C; ch++) {
            out_values[(sample_id * D * D + 0) * C + ch] += values[ch] * alpha * (a1 * a1 - con[0]);
            out_values[(sample_id * D * D + 1) * C + ch] += values[ch] * alpha * (a1 * a2 - con[1]);
            out_values[(sample_id * D * D + 2) * C + ch] += values[ch] * alpha * (a1 * a2 - con[1]);
            out_values[(sample_id * D * D + 3) * C + ch] += values[ch] * alpha * (a2 * a2 - con[3]);
        }
    }
}

void FORWARD::render(
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
    FLOAT* out_values)
{
    switch (function) {
        case CudaSampler::Function::gaussian:
            renderCUDA<gaussian> << <blocks, NUM_THREADS >> > (
                D, C,
                ranges,
                sample_ranges,
                point_list,
                sample_point_list,
                means,
                values,
                conics,
                samples,
                out_values);
            break;
        case CudaSampler::Function::derivative:
            renderCUDA<gaussian_derivative> << <blocks, NUM_THREADS >> > (
                D, C,
                ranges,
                sample_ranges,
                point_list,
                sample_point_list,
                means,
                values,
                conics,
                samples,
                out_values);
            break;
        case CudaSampler::Function::laplacian:
            renderCUDA<gaussian_laplacian> << <blocks, NUM_THREADS >> > (
                D, C,
                ranges,
                sample_ranges,
                point_list,
                sample_point_list,
                means,
                values,
                conics,
                samples,
                out_values);
            break;
    }
}

void FORWARD::preprocess(
    const int P, const int D, const int N, const int C,
    const FLOAT* means,
    const FLOAT* values,
    const FLOAT* covariances,
    const FLOAT* conics,
    FLOAT* radii,
    const int* grid,
    const FLOAT* grid_offset,
    uint32_t* tiles_touched)
{
	preprocessCUDA << <(P + 255) / 256, 256 >> > (
		P, D, N, C,
        means,
        values,
        covariances,
        conics,
        radii,
        grid,
        grid_offset,
		tiles_touched
    );
}
