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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define NUM_THREADS 256

// Backward version of the rendering procedure.
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
	const FLOAT* __restrict__ dL_dout_values,
	FLOAT* __restrict__ dL_dmeans,
	FLOAT* __restrict__ dL_dvalues,
	FLOAT* __restrict__ dL_dconics,
	FLOAT* __restrict__ dL_dopacities,
	FLOAT* __restrict__ dL_dsamples)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();

	const uint2 range = ranges[block.group_index().x];
	const int rounds = ((range.y - range.x + NUM_THREADS - 1) / NUM_THREADS);
	int toDo = range.y - range.x;

	const uint2 sample_range = sample_ranges[block.group_index().x];
    const int num_samples = sample_range.y - sample_range.x;
    const int samples_per_thread = (num_samples + NUM_THREADS - 1) / NUM_THREADS;
    const int sample_offset = block.thread_rank() * samples_per_thread;

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

            for (int s = 0; s < samples_per_thread; s++) {
                if (sample_offset + s >= num_samples)
                    break;

                const int sample_id = sample_point_list[sample_range.x + sample_offset + s];
                const FLOAT* sample = samples + sample_id * D;

                // Resample as before
                FLOAT power = 0.0;
                if (D == 2) {
                    for (int k = 0; k < D; k++)  X[k] = mean[k] - sample[k];
                    power = -0.5 * (con[0] * X[0] * X[0] + con[3] * X[1] * X[1]) - con[1] * X[0] * X[1];
                }

                if (power > 0.0)
                    continue;

                const FLOAT G = exp(power);
                const FLOAT opacity = opacities[id];

                FLOAT dL_dG = 0.0;
                for (int ch = 0; ch < C; ch++) {
                    const FLOAT dL_dchannel = dL_dout_values[sample_id * C + ch];
                    atomicAdd(&(dL_dvalues[id * C + ch]), G * opacity * dL_dchannel);
                    dL_dG += values[id * C + ch] * dL_dchannel;
                }

                if (D == 2) {
                    // Helpful reusable temporary variables
                    const FLOAT gdx = opacity * G * X[0];
                    const FLOAT gdy = opacity * G * X[1];
                    const FLOAT dG_ddelx = gdx * con[0] + gdy * con[1];
                    const FLOAT dG_ddely = gdx * con[1] + gdy * con[3];

                    atomicAdd(&(dL_dmeans[id * D + 0]), -dL_dG * dG_ddelx);
                    atomicAdd(&(dL_dmeans[id * D + 1]), -dL_dG * dG_ddely);

                    atomicAdd(&(dL_dsamples[sample_id * D + 0]), dL_dG * dG_ddelx);
                    atomicAdd(&(dL_dsamples[sample_id * D + 1]), dL_dG * dG_ddely);

                    atomicAdd(&(dL_dconics[id * D * D + 0]), -0.5 * gdx * X[0] * dL_dG);
                    atomicAdd(&(dL_dconics[id * D * D + 1]), -0.5 * gdy * X[0] * dL_dG);
                    atomicAdd(&(dL_dconics[id * D * D + 2]), -0.5 * gdx * X[1] * dL_dG);
                    atomicAdd(&(dL_dconics[id * D * D + 3]), -0.5 * gdy * X[1] * dL_dG);

                    atomicAdd(&(dL_dopacities[id]), G * dL_dG);
                }
            }
        }
    }

    delete X;
}

void BACKWARD::render(
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
    const FLOAT* dL_dout_values,
    FLOAT* dL_dmeans,
    FLOAT* dL_dvalues,
    FLOAT* dL_dconics,
    FLOAT* dL_dopacities,
    FLOAT* dL_dsamples)
{
	renderCUDA << <blocks, NUM_THREADS >> >(
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
		dL_dout_values,
		dL_dmeans,
		dL_dvalues,
		dL_dconics,
		dL_dopacities,
        dL_dsamples);
}
