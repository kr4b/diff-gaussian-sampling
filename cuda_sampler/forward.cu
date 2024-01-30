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
    const FLOAT* opacities,
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
    if (D == 2) {
        FLOAT det = (cov[0] * cov[3] - cov[1] * cov[1]);
        if (det == 0.0)
            return;
        FLOAT mid = 0.5f * (cov[0] + cov[3]);
        FLOAT lambda1 = mid + sqrt(max(0.1f, mid * mid - det)) / 2.0;
        my_radius = 3.0 * sqrt(lambda1);
        uint2 rect_min, rect_max;
        getRect(means + idx * D, my_radius, rect_min, rect_max, grid, grid_offset);
        touched = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
    } else {
        return;
    }

    if (touched == 0)
        return;

	// Store some useful helper data for the next steps.
	radii[idx] = my_radius;
	tiles_touched[idx] = touched;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
__global__ void renderCUDA(
    const int D, const int C,
	const uint2* __restrict__ ranges,
	const uint2* __restrict__ sample_ranges,
	const uint32_t* __restrict__ point_list,
	const uint32_t* __restrict__ sample_point_list,
	const FLOAT* __restrict__ means,
	const FLOAT* __restrict__ values,
	const FLOAT* __restrict__ conics,
	const FLOAT* __restrict__ opacities,
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

    FLOAT* X = new float[D];

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

            for (int s = 0; s < samples_per_thread; s++) {
                if (sample_offset + s >= num_samples)
                    break;

                const int sample_id = sample_point_list[sample_range.x + sample_offset + s];
                const FLOAT* sample = samples + sample_id * D;

                for (int k = 0; k < D; k++)  X[k] = mean[k] - sample[k];

                const FLOAT opacity = opacities[id];

                if (D == 2) {
                    const FLOAT power = -0.5 * (con[0] * X[0] * X[0] + con[3] * X[1] * X[1]) - con[1] * X[0] * X[1];
                    if (power > 0.0) return;

                    const FLOAT alpha = opacity * exp(power);
                    for (int ch = 0; ch < C; ch++)
                        out_values[sample_id * C + ch] += values[id * C + ch] * alpha;
                }
            }
        }
    }

    delete X;
}

void FORWARD::render(
    const int D, const int C,
    const int blocks,
    const uint2* ranges,
    const uint2* sample_ranges,
    const uint32_t* point_list,
    const uint32_t* sample_point_list,
    const FLOAT* means,
    const FLOAT* values,
    const FLOAT* conics,
    const FLOAT* opacities,
	const FLOAT* samples,
    FLOAT* out_value)
{
	renderCUDA << <blocks, NUM_THREADS >> > (
        D, C,
        ranges,
        sample_ranges,
		point_list,
		sample_point_list,
		means,
		values,
		conics,
		opacities,
        samples,
		out_value);
}

void FORWARD::preprocess(
    const int P, const int D, const int N, const int C,
    const FLOAT* means,
    const FLOAT* values,
    const FLOAT* covariances,
    const FLOAT* conics,
    const FLOAT* opacities,
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
        opacities,
        radii,
        grid,
        grid_offset,
		tiles_touched
    );
}
