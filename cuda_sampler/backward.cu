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

typedef void(*gaussian_func)(const FLOAT*, const FLOAT*, const FLOAT*, const FLOAT*, FLOAT*, FLOAT*, FLOAT*, int, int, int);

// Backward version of the rendering procedure.
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
	const FLOAT* __restrict__ dL_dout_values,
	FLOAT* __restrict__ dL_dmeans,
	FLOAT* __restrict__ dL_dvalues,
	FLOAT* __restrict__ dL_dconics)
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
            const FLOAT* value = values + id * C;

            for (int s = 0; s < samples_per_thread; s++) {
                if (sample_offset + s >= num_samples)
                    break;

                const int sample_id = sample_point_list[sample_range.x + sample_offset + s];
                const FLOAT* sample = samples + sample_id * D;

                for (int k = 0; k < D; k++)  X[k] = mean[k] - sample[k];

                F(X, con, value, dL_dout_values, dL_dmeans + id * D, dL_dvalues + id * C, dL_dconics + id * D * D, sample_id, D, C);
            }
        }
    }

    delete X;
}

__forceinline__ __device__ void gaussian(
    const FLOAT* X, const FLOAT* con, const FLOAT* values, const FLOAT* dL_dout_values,
    FLOAT* dL_dmeans, FLOAT* dL_dvalues, FLOAT* dL_dconics, int sample_id, int D, int C)
{
    if (D == 1) {
        FLOAT power = -0.5 * con[0] * X[0] * X[0];
        if (power > 0.0) return;

        const FLOAT G = exp(power);

        FLOAT dL_dG = 0.0;
        for (int ch = 0; ch < C; ch++) {
            const FLOAT dL_dchannel = dL_dout_values[sample_id * C + ch];
            atomicAdd(dL_dvalues + ch, G * dL_dchannel);
            dL_dG += values[ch] * dL_dchannel;
        }

        const FLOAT gdx = G * X[0];
        const FLOAT dG_ddelx = gdx * con[0];
        const FLOAT dL_dx = dL_dG * dG_ddelx;

        atomicAdd(dL_dmeans + 0, -dL_dx);
        atomicAdd(dL_dconics + 0, -0.5 * gdx * X[0] * dL_dG);
    } else if (D == 2) {
        FLOAT power = -0.5 * (con[0] * X[0] * X[0] + con[3] * X[1] * X[1]) - con[1] * X[0] * X[1];
        if (power > 0.0) return;

        const FLOAT G = exp(power);

        FLOAT dL_dG = 0.0;
        for (int ch = 0; ch < C; ch++) {
            const FLOAT dL_dchannel = dL_dout_values[sample_id * C + ch];
            atomicAdd(dL_dvalues + ch, G * dL_dchannel);
            dL_dG += values[ch] * dL_dchannel;
        }

        const FLOAT gdx = G * X[0];
        const FLOAT gdy = G * X[1];
        const FLOAT dG_ddelx = gdx * con[0] + gdy * con[1];
        const FLOAT dG_ddely = gdx * con[1] + gdy * con[3];
        const FLOAT dL_dx = dL_dG * dG_ddelx;
        const FLOAT dL_dy = dL_dG * dG_ddely;

        atomicAdd(dL_dmeans + 0, -dL_dx);
        atomicAdd(dL_dmeans + 1, -dL_dy);

        atomicAdd(dL_dconics + 0, -0.5 * gdx * X[0] * dL_dG);
        atomicAdd(dL_dconics + 1, -gdy * X[0] * dL_dG);
        // atomicAdd(dL_dconics + 2, -gdx * X[1] * dL_dG);
        atomicAdd(dL_dconics + 3, -0.5 * gdy * X[1] * dL_dG);
    }
}

__forceinline__ __device__ void gaussian_derivative(
    const FLOAT* X, const FLOAT* con, const FLOAT* values, const FLOAT* dL_dout_values,
    FLOAT* dL_dmeans, FLOAT* dL_dvalues, FLOAT* dL_dconics, int sample_id, int D, int C)
{
    if (D == 1) {
        const FLOAT x1 = con[0] * X[0];
        FLOAT power = -0.5 * x1 * X[0];
        if (power > 0.0) return;

        const FLOAT G = exp(power);

        FLOAT dL_dG = 0.0;
        for (int ch = 0; ch < C; ch++) {
            const FLOAT dL_dchannel = dL_dout_values[(sample_id * D + 0) * C + ch];
            atomicAdd(dL_dvalues + ch, x1 * dL_dchannel * G);
            dL_dG += values[ch] * dL_dchannel;
        }

        const FLOAT dL_dx = (x1 * x1 - con[0]) * dL_dG * G;

        atomicAdd(dL_dmeans + 0, -dL_dx);
        atomicAdd(dL_dconics + 0, (X[0] - 0.5 * X[0] * X[0] * x1) * dL_dG * G);
    } else if (D == 2) {
        const FLOAT x1 = con[0] * X[0];
        const FLOAT x2 = con[3] * X[1];
        FLOAT power = -0.5 * (x1 * X[0] + x2 * X[1]) - con[1] * X[0] * X[1];
        if (power > 0.0) return;

        const FLOAT G = exp(power);
        const FLOAT a1 = x1 + con[1] * X[1];
        const FLOAT a2 = x2 + con[1] * X[0];

        FLOAT dL_dGx = 0.0;
        FLOAT dL_dGy = 0.0;
        for (int ch = 0; ch < C; ch++) {
            const FLOAT dL_dchannelx = dL_dout_values[(sample_id * D + 0) * C + ch];
            const FLOAT dL_dchannely = dL_dout_values[(sample_id * D + 1) * C + ch];
            const FLOAT gx = a1 * dL_dchannelx + a2 * dL_dchannely;
            atomicAdd(dL_dvalues + ch, gx * G);
            dL_dGx += values[ch] * dL_dchannelx;
            dL_dGy += values[ch] * dL_dchannely;
        }

        const FLOAT gx = (a1 * dL_dGx + a2 * dL_dGy);
        const FLOAT dL_dx = ((a1 * a1 - con[0]) * dL_dGx + (a1 * a2 - con[1]) * dL_dGy) * G;
        const FLOAT dL_dy = ((a2 * a2 - con[3]) * dL_dGy + (a1 * a2 - con[1]) * dL_dGx) * G;

        atomicAdd(dL_dmeans + 0, -dL_dx);
        atomicAdd(dL_dmeans + 1, -dL_dy);

        atomicAdd(dL_dconics + 0, (X[0] * dL_dGx - 0.5 * X[0] * X[0] * gx) * G);
        atomicAdd(dL_dconics + 1, (X[1] * dL_dGx + X[0] * dL_dGy - X[0] * X[1] * gx) * G);
        // atomicAdd(dL_dconics + 2, (X[0] * dL_dGy + X[1] * dL_dGx - X[1] * X[0] * gx) * G);
        atomicAdd(dL_dconics + 3, (X[1] * dL_dGy - 0.5 * X[1] * X[1] * gx) * G);
    }
}

__forceinline__ __device__ void gaussian_laplacian(
    const FLOAT* X, const FLOAT* con, const FLOAT* values, const FLOAT* dL_dout_values,
    FLOAT* dL_dmeans, FLOAT* dL_dvalues, FLOAT* dL_dconics, int sample_id, int D, int C)
{
    if (D == 1) {
        const FLOAT x1 = con[0] * X[0];
        FLOAT power = -0.5 * x1 * X[0];
        if (power > 0.0) return;

        const FLOAT G = exp(power);

        FLOAT dL_dG = 0.0;
        for (int ch = 0; ch < C; ch++) {
            const FLOAT dL_dchannel = dL_dout_values[(sample_id * D * D + 0) * C + ch];
            const FLOAT gx = (x1 * x1 - con[0]) * dL_dchannel;
            atomicAdd(dL_dvalues + ch, gx * G);
            dL_dG += values[ch] * dL_dchannel;
        }

        const FLOAT dL_dx = (x1 * x1 * x1 - 3.0 * con[0] * x1) * dL_dG * G;
        const FLOAT dV_dc = (2.0 * x1 * X[0]
                          -  0.5 * (x1 * x1 - con[0]) * X[0] * X[0]
                          -  1.0) * dL_dG * G;

        atomicAdd(dL_dmeans + 0, -dL_dx);
        atomicAdd(dL_dconics + 0, dV_dc);
    } else if (D == 2) {
        const FLOAT x1 = con[0] * X[0];
        const FLOAT x2 = con[3] * X[1];
        FLOAT power = -0.5 * (x1 * X[0] + x2 * X[1]) - con[1] * X[0] * X[1];
        if (power > 0.0) return;

        const FLOAT G = exp(power);
        const FLOAT a1 = x1 + con[1] * X[1];
        const FLOAT a2 = x2 + con[1] * X[0];

        const FLOAT dxx = a1 * a1 - con[0];
        const FLOAT dxy = a1 * a2 - con[1];
        const FLOAT dyy = a2 * a2 - con[3];

        FLOAT dL_dGxx = 0.0;
        FLOAT dL_dGxy = 0.0;
        FLOAT dL_dGyx = 0.0;
        FLOAT dL_dGyy = 0.0;
        for (int ch = 0; ch < C; ch++) {
            const FLOAT dL_dchannelxx = dL_dout_values[(sample_id * D * D + 0) * C + ch];
            const FLOAT dL_dchannelxy = dL_dout_values[(sample_id * D * D + 1) * C + ch];
            const FLOAT dL_dchannelyx = dL_dout_values[(sample_id * D * D + 2) * C + ch];
            const FLOAT dL_dchannelyy = dL_dout_values[(sample_id * D * D + 3) * C + ch];
            const FLOAT gxx = dxx * dL_dchannelxx + dxy * dL_dchannelxy
                            + dxy * dL_dchannelyx + dyy * dL_dchannelyy;
            atomicAdd(dL_dvalues + ch, gxx * G);
            dL_dGxx += values[ch] * dL_dchannelxx;
            dL_dGxy += values[ch] * dL_dchannelxy;
            dL_dGyx += values[ch] * dL_dchannelyx;
            dL_dGyy += values[ch] * dL_dchannelyy;
        }

        const FLOAT dL_dx = ((a1 * a1 * a1 - 3.0 * con[0] * a1) * dL_dGxx
                          +  (a1 * a2 * a1 - con[1] * a1 - (con[1] * a1 + con[0] * a2)) * (dL_dGxy + dL_dGyx)
                          +  (a2 * a2 * a1 - con[3] * a1 - 2.0 * con[1] * a2) * dL_dGyy
                          ) * G;
        const FLOAT dL_dy = ((a1 * a1 * a2 - con[0] * a2 - 2.0 * con[1] * a1) * dL_dGxx
                          +  (a1 * a2 * a2 - con[1] * a2 - (con[3] * a1 + con[1] * a2)) * (dL_dGxy + dL_dGyx)
                          +  (a2 * a2 * a2 - 3.0 * con[3] * a2) * dL_dGyy
                          ) * G;

        atomicAdd(dL_dmeans + 0, -dL_dx);
        atomicAdd(dL_dmeans + 1, -dL_dy);

        const FLOAT dVxx_dcxx = -0.5 * dxx * X[0] * X[0] + 2.0 * a1 * X[0] - 1.0;
        const FLOAT dVxy_dcxx = -0.5 * dxy * X[0] * X[0] + a2 * X[0];
        const FLOAT dVyy_dcxx = -0.5 * dyy * X[0] * X[0];

        const FLOAT dVxx_dcxy = -dxx * X[0] * X[1] + 2.0 * a1 * X[1];
        const FLOAT dVxy_dcxy = -dxy * X[0] * X[1] + a2 * X[1] + a1 * X[0] - 1.0;
        const FLOAT dVyy_dcxy = -dyy * X[0] * X[1] + 2.0 * a2 * X[0];

        const FLOAT dVxx_dcyy = -0.5 * dxx * X[1] * X[1];
        const FLOAT dVxy_dcyy = -0.5 * dxy * X[1] * X[1] + a1 * X[1];
        const FLOAT dVyy_dcyy = -0.5 * dyy * X[1] * X[1] + 2.0 * a2 * X[1] - 1.0;

        atomicAdd(dL_dconics + 0, (dVxx_dcxx * dL_dGxx + dVxy_dcxx * (dL_dGxy + dL_dGyx) + dVyy_dcxx * dL_dGyy) * G);
        atomicAdd(dL_dconics + 1, (dVxx_dcxy * dL_dGxx + dVxy_dcxy * (dL_dGxy + dL_dGyx) + dVyy_dcxy * dL_dGyy) * G);
        // atomicAdd(dL_dconics + 2, (dVxx_dcxy * dL_dGxx + dVxy_dcxy * (dL_dGxy + dL_dGyx) + dVyy_dcxy * dL_dGyy) * G);
        atomicAdd(dL_dconics + 3, (dVxx_dcyy * dL_dGxx + dVxy_dcyy * (dL_dGxy + dL_dGyx) + dVyy_dcyy * dL_dGyy) * G);
    }
}

__forceinline__ __device__ void gaussian_third(
    const FLOAT* X, const FLOAT* con, const FLOAT* values, const FLOAT* dL_dout_values,
    FLOAT* dL_dmeans, FLOAT* dL_dvalues, FLOAT* dL_dconics, int sample_id, int D, int C)
{
    if (D == 1) {
        const FLOAT x1 = con[0] * X[0];
        FLOAT power = -0.5 * x1 * X[0];
        if (power > 0.0) return;

        const FLOAT G = exp(power);

        FLOAT dL_dG = 0.0;
        for (int ch = 0; ch < C; ch++) {
            const FLOAT dL_dchannel = dL_dout_values[(sample_id * D * D * D + 0) * C + ch];
            const FLOAT gx = (3.0 * con[0] * x1 - x1 * x1 * x1) * dL_dchannel;
            atomicAdd(dL_dvalues + ch, gx * G);
            dL_dG += values[ch] * dL_dchannel;
        }

        const FLOAT dL_dx = (6.0 * con[0] * x1 * x1
                          -  x1 * x1 * x1 * x1
                          -  3.0 * con[0] * con[0]) * dL_dG * G;
        const FLOAT dV_dc = (2.0 * X[0] * X[0]
                          -  2.0 * x1 * x1 * X[0]
                          -  0.5 * (2.0 * X[0] * x1 - X[0]) * X[0] * X[0]
                          +  0.5 * (x1 * x1 - con[0]) * x1 * X[0] * X[0]) * dL_dG * G;

        atomicAdd(dL_dmeans + 0, -dL_dx);
        atomicAdd(dL_dconics + 0, dV_dc);
    } else if (D == 2) {
        const FLOAT x1 = con[0] * X[0];
        const FLOAT x2 = con[3] * X[1];
        FLOAT power = -0.5 * (x1 * X[0] + x2 * X[1]) - con[1] * X[0] * X[1];
        if (power > 0.0) return;

        const FLOAT G = exp(power);
        const FLOAT a1 = x1 + con[1] * X[1];
        const FLOAT a2 = x2 + con[1] * X[0];

        const FLOAT dxxx = 3.0 * con[0] * a1 - a1 * a1 * a1;
        const FLOAT dxxy = 2.0 * con[1] * a1 - a1 * a1 * a2 + con[0] * a2;
        const FLOAT dxyy = 2.0 * con[1] * a2 - a1 * a2 * a2 + con[3] * a1;
        const FLOAT dyyy = 3.0 * con[3] * a2 - a2 * a2 * a2;

        FLOAT dL_dGxxx = 0.0;
        FLOAT dL_dGxxy = 0.0;
        FLOAT dL_dGxyx = 0.0;
        FLOAT dL_dGxyy = 0.0;
        FLOAT dL_dGyxx = 0.0;
        FLOAT dL_dGyxy = 0.0;
        FLOAT dL_dGyyx = 0.0;
        FLOAT dL_dGyyy = 0.0;

        for (int ch = 0; ch < C; ch++) {
            const FLOAT dL_dchannelxxx = dL_dout_values[(sample_id * D * D * D + 0) * C + ch];
            const FLOAT dL_dchannelxxy = dL_dout_values[(sample_id * D * D * D + 1) * C + ch];
            const FLOAT dL_dchannelxyx = dL_dout_values[(sample_id * D * D * D + 2) * C + ch];
            const FLOAT dL_dchannelxyy = dL_dout_values[(sample_id * D * D * D + 3) * C + ch];
            const FLOAT dL_dchannelyxx = dL_dout_values[(sample_id * D * D * D + 4) * C + ch];
            const FLOAT dL_dchannelyxy = dL_dout_values[(sample_id * D * D * D + 5) * C + ch];
            const FLOAT dL_dchannelyyx = dL_dout_values[(sample_id * D * D * D + 6) * C + ch];
            const FLOAT dL_dchannelyyy = dL_dout_values[(sample_id * D * D * D + 7) * C + ch];
            const FLOAT gxx = dxxx * dL_dchannelxxx + dxxy * dL_dchannelxxy
                            + dxxy * dL_dchannelxyx + dxyy * dL_dchannelxyy
                            + dxxy * dL_dchannelyxx + dxyy * dL_dchannelyxy
                            + dxyy * dL_dchannelyyx + dyyy * dL_dchannelyyy;
            atomicAdd(dL_dvalues + ch, gxx * G);
            dL_dGxxx += values[ch] * dL_dchannelxxx;
            dL_dGxxy += values[ch] * dL_dchannelxxy;
            dL_dGxyx += values[ch] * dL_dchannelxyx;
            dL_dGxyy += values[ch] * dL_dchannelxyy;
            dL_dGyxx += values[ch] * dL_dchannelyxx;
            dL_dGyxy += values[ch] * dL_dchannelyxy;
            dL_dGyyx += values[ch] * dL_dchannelyyx;
            dL_dGyyy += values[ch] * dL_dchannelyyy;
        }

        const FLOAT dxxy_dx = 2.0 * a1 * a2 * con[0] + a1 * a1 * con[1] - 3.0 * con[0] * con[1];
        const FLOAT dxyy_dx = 2.0 * a1 * a2 * con[1] + a2 * a2 * con[0] - con[3] * con[0] - 2.0 * con[1] * con[1];
        const FLOAT dL_dx = ((dxxx * a1 - 3.0 * con[0] * con[0] + 3.0 * a1 * a1 * con[0]) * dL_dGxxx
                          +  (dxxy * a1 + dxxy_dx) * dL_dGxxy + (dxxy * a1 + dxxy_dx) * dL_dGxyx
                          +  (dxyy * a1 + dxyy_dx) * dL_dGxyy + (dxxy * a1 + dxxy_dx) * dL_dGyxx
                          +  (dxyy * a1 + dxyy_dx) * dL_dGyxy + (dxyy * a1 + dxyy_dx) * dL_dGyyx
                          +  (dyyy * a1 - 3.0 * con[3] * con[1] + 3.0 * a2 * a2 * con[1]) * dL_dGyyy
                          ) * G;
        const FLOAT dxxy_dy = 2.0 * a1 * a2 * con[1] + a1 * a1 * con[3] - con[0] * con[3] - 2.0 * con[1] * con[1];
        const FLOAT dxyy_dy = 2.0 * a1 * a2 * con[3] + a2 * a2 * con[1] - 3.0 * con[3] * con[1];
        const FLOAT dL_dy = ((dxxx * a2 - 3.0 * con[0] * con[1] + 3.0 * a1 * a1 * con[1]) * dL_dGxxx
                          +  (dxxy * a2 + dxxy_dy) * dL_dGxxy + (dxxy * a2 + dxxy_dy) * dL_dGxyx
                          +  (dxyy * a2 + dxyy_dy) * dL_dGxyy + (dxxy * a2 + dxxy_dy) * dL_dGyxx
                          +  (dxyy * a2 + dxyy_dy) * dL_dGyxy + (dxyy * a2 + dxyy_dy) * dL_dGyyx
                          +  (dyyy * a2 - 3.0 * con[3] * con[3] + 3.0 * a2 * a2 * con[3]) * dL_dGyyy
                          ) * G;

        atomicAdd(dL_dmeans + 0, -dL_dx);
        atomicAdd(dL_dmeans + 1, -dL_dy);

        const FLOAT dVxxx_dcxx = -0.5 * dxxx * X[0] * X[0] + 3.0 * con[0] * X[0] + 3.0 * a1 - 3.0 * a1 * a1 * X[0];
        const FLOAT dVxxy_dcxx = -0.5 * dxxy * X[0] * X[0] + 2.0 * con[1] * X[0] - 2.0 * a1 * a2 * X[0] + a2;
        const FLOAT dVxyy_dcxx = -0.5 * dxyy * X[0] * X[0] - a2 * a2 * X[0] + con[3] * X[0];
        const FLOAT dVyyy_dcxx = -0.5 * dyyy * X[0] * X[0];

        const FLOAT dVxxx_dcxy = -dxxx * X[0] * X[1] + 3.0 * con[0] * X[1] - 3.0 * a1 * a1 * X[1];
        const FLOAT dVxxy_dcxy = -dxxy * X[0] * X[1] + 2.0 * con[1] * X[1] + 2.0 * a1 - 2.0 * a1 * a2 * X[1] - a1 * a1 * X[0] + con[0] * X[0];
        const FLOAT dVxyy_dcxy = -dxyy * X[0] * X[1] + 2.0 * con[1] * X[0] + 2.0 * a2 - a2 * a2 * X[1] - 2.0 * a1 * a2 * X[0] + con[3] * X[1];
        const FLOAT dVyyy_dcxy = -dyyy * X[0] * X[1] + 3.0 * con[3] * X[0] - 3.0 * a2 * a2 * X[0];

        const FLOAT dVxxx_dcyy = -0.5 * dxxx * X[1] * X[1];
        const FLOAT dVxxy_dcyy = -0.5 * dxxy * X[1] * X[1] - a1 * a1 * X[1] + con[0] * X[1];
        const FLOAT dVxyy_dcyy = -0.5 * dxyy * X[1] * X[1] + 2.0 * con[1] * X[1] - 2.0 * a1 * a2 * X[1] + a1;
        const FLOAT dVyyy_dcyy = -0.5 * dyyy * X[1] * X[1] + 3.0 * con[3] * X[1] + 3.0 * a2 - 3.0 * a2 * a2 * X[1];

        atomicAdd(dL_dconics + 0, (dVxxx_dcxx * dL_dGxxx + dVxxy_dcxx * (dL_dGxxy + dL_dGxyx + dL_dGyxx) + dVxyy_dcxx * (dL_dGxyy + dL_dGyxy + dL_dGyyx) + dVyyy_dcxx * dL_dGyyy) * G);
        atomicAdd(dL_dconics + 1, (dVxxx_dcxy * dL_dGxxx + dVxxy_dcxy * (dL_dGxxy + dL_dGxyx + dL_dGyxx) + dVxyy_dcxy * (dL_dGxyy + dL_dGyxy + dL_dGyyx) + dVyyy_dcxy * dL_dGyyy) * G);
        // atomicAdd(dL_dconics + 2, (dVxxx_dcxy * dL_dGxxx + dVxxy_dcxy * (dL_dGxxy + dL_dGxyx + dL_dGyxx) + dVxyy_dcxy * (dL_dGxyy + dL_dGyxy + dL_dGyyx) + dVyyy_dcxy * dL_dGyyy) * G);
        atomicAdd(dL_dconics + 3, (dVxxx_dcyy * dL_dGxxx + dVxxy_dcyy * (dL_dGxxy + dL_dGxyx + dL_dGyxx) + dVxyy_dcyy * (dL_dGxyy + dL_dGyxy + dL_dGyyx) + dVyyy_dcyy * dL_dGyyy) * G);
    }
}

void BACKWARD::render(
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
    FLOAT* dL_dconics)
{
    switch (function) {
        case CudaSampler::Function::gaussian:
            renderCUDA<gaussian> << <blocks, NUM_THREADS >> >(
                D, C,
                ranges,
                sample_ranges,
                point_list,
                sample_point_list,
                means,
                values,
                conics,
                samples,
                dL_dout_values,
                dL_dmeans,
                dL_dvalues,
                dL_dconics);
            break;
        case CudaSampler::Function::derivative:
            renderCUDA<gaussian_derivative> << <blocks, NUM_THREADS >> >(
                D, C,
                ranges,
                sample_ranges,
                point_list,
                sample_point_list,
                means,
                values,
                conics,
                samples,
                dL_dout_values,
                dL_dmeans,
                dL_dvalues,
                dL_dconics);
            break;
        case CudaSampler::Function::laplacian:
            renderCUDA<gaussian_laplacian> << <blocks, NUM_THREADS >> >(
                D, C,
                ranges,
                sample_ranges,
                point_list,
                sample_point_list,
                means,
                values,
                conics,
                samples,
                dL_dout_values,
                dL_dmeans,
                dL_dvalues,
                dL_dconics);
            break;
        case CudaSampler::Function::third:
            renderCUDA<gaussian_third> << <blocks, NUM_THREADS >> >(
                D, C,
                ranges,
                sample_ranges,
                point_list,
                sample_point_list,
                means,
                values,
                conics,
                samples,
                dL_dout_values,
                dL_dmeans,
                dL_dvalues,
                dL_dconics);
            break;
    }
}
