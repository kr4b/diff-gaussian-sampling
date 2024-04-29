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

#include "sampler_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1) {
        step /= 2;
        if (n >> msb) msb += step;
        else msb -= step;
    }
    if (n >> msb) msb++;
    return msb;
}

// Generates one key/value pair for all Gaussian / tile overlaps.
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
    int P, int D,
    const FLOAT* means,
    const uint32_t* offsets,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    const FLOAT* radii,
    const int* grid,
    const FLOAT* grid_offset)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Generate no key/value pair for invisible Gaussians
    if (radii[idx] > 0.0f) {
        // Find this Gaussian's offset in buffer for writing keys/values.
        uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
        uint* rect_min = new uint[D];
        uint* rect_max = new uint[D];

        getRect(D, means + idx * D, radii[idx], rect_min, rect_max, grid, grid_offset);

        // For each tile that the bounding rect overlaps, emit a
        // key/value pair. The key is |  tile ID  |      idx      |,
        // and the value is the ID of the Gaussian. Sorting the values
        // with this key yields Gaussian IDs in a list, such that they
        // are sorted by tile.
        if (D == 1) {
            for (int x = rect_min[0]; x < rect_max[0]; x++) {
                uint64_t key = x;
                key <<= 32;
                key |= (uint32_t) idx;
                gaussian_keys_unsorted[off] = key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        } else if (D == 2) {
            for (int y = rect_min[1]; y < rect_max[1]; y++) {
                for (int x = rect_min[0]; x < rect_max[0]; x++) {
                    uint64_t key = y * grid[0] + x;
                    key <<= 32;
                    key |= (uint32_t) idx;
                    gaussian_keys_unsorted[off] = key;
                    gaussian_values_unsorted[off] = idx;
                    off++;
                }
            }
        }

        delete rect_min;
        delete rect_max;
    }
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L) return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    uint64_t key = point_list_keys[idx];
    uint32_t currtile = key >> 32;
    if (idx == 0) {
        ranges[currtile].x = 0;
    } else {
        uint32_t prevtile = point_list_keys[idx - 1] >> 32;
        if (currtile != prevtile) {
            ranges[prevtile].y = idx;
            ranges[currtile].x = idx;
        }
    }
    if (idx == L - 1) ranges[currtile].y = L;
}

// Generates one key/value pair for all sample / tile overlaps.
// Run once per sample (1:N mapping).
__global__ void sampleWithKeys(
    int N, int D,
    const FLOAT* means,
    uint64_t* sample_keys_unsorted,
    uint32_t* sample_values_unsorted,
    const int* grid,
    const FLOAT* grid_offset)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= N)
        return;

    uint* tile = new uint[D];
	for (size_t i = 0; i < D; i++) {
		tile[i] = (uint)min(grid[i], max((int)0, (int)((means[idx * D + i] - grid_offset[i]) / BLOCK_SIZE)));
	}

    // For the tile that the sample resides in, emit a key/value pair.
    // The key is |  tile ID  |      idx      |,
    // and the value is the ID of the sample. Sorting the values
    // with this key yields sample IDs in a list, such that they
    // are sorted by tile.
    uint64_t key;
    if (D == 1) {
        key = tile[0];
    } else if (D == 2) {
        key = tile[1] * grid[0] + tile[0];
    }
    key <<= 32;
    key |= (uint32_t) idx;
    sample_keys_unsorted[idx] = key;
    sample_values_unsorted[idx] = idx;

    delete tile;
}

CudaSampler::GeometryState CudaSampler::GeometryState::fromChunk(char*& chunk, size_t P) {
    GeometryState geom;
    obtain(chunk, geom.tiles_touched, P, 128);
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
    obtain(chunk, geom.scanning_space, geom.scan_size, 128);
    obtain(chunk, geom.point_offsets, P, 128);
    return geom;
}

CudaSampler::BinningState CudaSampler::BinningState::fromChunk(char*& chunk, size_t P) {
    BinningState binning;
    obtain(chunk, binning.point_list, P, 128);
    obtain(chunk, binning.point_list_unsorted, P, 128);
    obtain(chunk, binning.point_list_keys, P, 128);
    obtain(chunk, binning.point_list_keys_unsorted, P, 128);
    cub::DeviceRadixSort::SortPairs(
        nullptr, binning.sorting_size,
        binning.point_list_keys_unsorted, binning.point_list_keys,
        binning.point_list_unsorted, binning.point_list, P);
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
    return binning;
}

// Forward rendering procedure for differentiable sampling
// of Gaussians.
int CudaSampler::Sampler::preprocess(
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
    const FLOAT* samples,
    uint2* ranges,
    uint2* sample_ranges,
    FLOAT* radii,
    bool debug)
{
    size_t chunk_size = required<GeometryState>(P);
    char* chunkptr = geometry_buffer(chunk_size);
    GeometryState geom_state = GeometryState::fromChunk(chunkptr, P);

    // Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
    CHECK_CUDA(FORWARD::preprocess(
        P, D, N, C,
        means,
        values,
        covariances,
        conics,
        radii,
        tile_grid,
        grid_offset,
        geom_state.tiles_touched
    ), debug)

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geom_state.scanning_space, geom_state.scan_size, geom_state.tiles_touched, geom_state.point_offsets, P), debug)

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geom_state.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr = binning_buffer(binning_chunk_size);
    BinningState binning_state = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each instance to be rendered, produce adequate [ tile | idx ] key
    // and corresponding dublicated Gaussian indices to be sorted
    duplicateWithKeys << <(P + 255) / 256, 256 >> > (
        P, D,
        means,
        geom_state.point_offsets,
        binning_state.point_list_keys_unsorted,
        binning_state.point_list_unsorted,
        radii,
        tile_grid,
        grid_offset)
    CHECK_CUDA(, debug)

    int bit = getHigherMsb(blocks);

    // Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        binning_state.list_sorting_space,
        binning_state.sorting_size, binning_state.point_list_keys_unsorted,
        binning_state.point_list_keys, binning_state.point_list_unsorted,
        binning_state.point_list, num_rendered, 0, 32 + bit), debug)

    // Identify start and end of per-tile workloads in sorted list
    if (num_rendered > 0) {
        identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
            num_rendered,
            binning_state.point_list_keys,
            ranges)
        CHECK_CUDA(, debug)
    }

    size_t sample_binning_chunk_size = required<BinningState>(N);
    char* sample_binning_chunkptr = sample_binning_buffer(sample_binning_chunk_size);
    BinningState sample_binning_state = BinningState::fromChunk(sample_binning_chunkptr, N);

    // For each point to be sampled, produce adequate [ tile | idx ] key
    // and corresponding sample indices to be sorted
    sampleWithKeys << <(N + 255) / 256, 256 >> > (
        N, D,
        samples,
        sample_binning_state.point_list_keys_unsorted,
        sample_binning_state.point_list_unsorted,
        tile_grid,
        grid_offset)
    CHECK_CUDA(, debug)

    // uint64_t* keys = new uint64_t[N];
    // cudaMemcpy(keys, sample_binning_state.point_list_keys_unsorted, N, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < N; i++) {
    //     std::cout << (keys[i] >> 32) << std::endl;
    // }

    // Sort complete list of samples indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        sample_binning_state.list_sorting_space,
        sample_binning_state.sorting_size, sample_binning_state.point_list_keys_unsorted,
        sample_binning_state.point_list_keys, sample_binning_state.point_list_unsorted,
        sample_binning_state.point_list, N, 0, 32 + bit), debug)

    // Identify start and end of per-tile workloads in sorted list
    identifyTileRanges << <(N + 255) / 256, 256 >> > (
        N,
        sample_binning_state.point_list_keys,
        sample_ranges)
    CHECK_CUDA(, debug)

    return num_rendered;
}

// Perform rendering using preprocessed data structures
void CudaSampler::Sampler::forward(
    const int P, const int D, const int N, const int C,
    const int blocks, const int num_rendered,
    const CudaSampler::Function function,
    const FLOAT* means,
    const FLOAT* values,
    const FLOAT* conics,
    const FLOAT* samples,
    char* binning_buffer,
    char* sample_binning_buffer,
    const uint2* ranges,
    const uint2* sample_ranges,
    FLOAT* out_values,
    bool debug)
{
    BinningState binning_state = BinningState::fromChunk(binning_buffer, num_rendered);
    BinningState sample_binning_state = BinningState::fromChunk(sample_binning_buffer, N);

    // Let each tile blend its range of Gaussians independently in parallel
    CHECK_CUDA(FORWARD::render(
        D, C, blocks,
        function,
        ranges,
        sample_ranges,
        binning_state.point_list,
        sample_binning_state.point_list,
        means,
        values,
        conics,
        samples,
        out_values), debug)
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaSampler::Sampler::backward(
    const int P, const int D, const int N, const int C,
    const int blocks, const int num_rendered,
    const CudaSampler::Function function,
    const FLOAT* means,
    const FLOAT* values,
    const FLOAT* conics,
    const FLOAT* samples,
    char* binning_buffer,
    char* sample_binning_buffer,
    const uint2* ranges,
	const uint2* sample_ranges,
    const FLOAT* dL_dout_values,
    FLOAT* dL_dmeans,
    FLOAT* dL_dvalues,
    FLOAT* dL_dconics,
    bool debug)
{
    BinningState binning_state = BinningState::fromChunk(binning_buffer, num_rendered);
    BinningState sample_binning_state = BinningState::fromChunk(sample_binning_buffer, N);

    // Compute loss gradients w.r.t. means, values, conics and samples.
    CHECK_CUDA(BACKWARD::render(
        D, C, blocks,
        function,
        ranges,
        sample_ranges,
        binning_state.point_list,
        sample_binning_state.point_list,
        means,
        values,
        conics,
        samples,
        dL_dout_values,
        dL_dmeans,
        dL_dvalues,
        dL_dconics), debug)
}

