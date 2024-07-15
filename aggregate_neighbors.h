/*
 * The following code is not in any way a modified version of the diff-gaussian-rasterization. 
 * Therefore, it is not restricted by the same license.
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> AggregateNeighborsPreprocessCUDA(
    const torch::Tensor& means,
    const torch::Tensor& conics,
    const torch::Tensor& radii,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> AggregateNeighborsCUDA(
    const torch::Tensor& features,
    const torch::Tensor& transform,
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& frequencies,
    const torch::Tensor& distance_transform,
    const torch::Tensor& indices,
    const torch::Tensor& ranges,
    const torch::Tensor& dists,
    const torch::Tensor& densities,
    const torch::Tensor& inv_total_densities,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> AggregateNeighborsBackwardCUDA(
    const torch::Tensor& features,
    const torch::Tensor& transform,
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& frequencies,
    const torch::Tensor& distance_transform,
    const torch::Tensor& indices,
    const torch::Tensor& ranges,
    const torch::Tensor& dists,
    const torch::Tensor& densities,
    const torch::Tensor& weights,
    const torch::Tensor& embeddings,
    const torch::Tensor& factors,
    const torch::Tensor& inv_total_densities,
    const torch::Tensor& dL_dneighbor_features,
    const bool debug);
