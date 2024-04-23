#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> AggregateNeighborsCUDA(
    const torch::Tensor& means,
    const torch::Tensor& radii,
    const torch::Tensor& features,
    const torch::Tensor& distance_transforms,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor> AggregateNeighborsBackwardCUDA(
    const torch::Tensor& features,
    const torch::Tensor& indices,
    const torch::Tensor& dists,
    const torch::Tensor& factors,
    const torch::Tensor& dL_dneighbor_features,
    const bool debug);
