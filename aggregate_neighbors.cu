#include "cuda_sampler/auxiliary.h"

#include <math.h>
#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Find all colliding Gaussians based on bounding circles approximation
// TODO: Optimize using a tree structure
__global__ void findCollisions(
    const int P, const int D, const int L, const int E,
    const FLOAT* means,
    const FLOAT* radii,
    const FLOAT* features,
    const FLOAT* distance_transforms,
    bool* indices,
    FLOAT* dists,
    FLOAT* factors,
    FLOAT* neighbor_features)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    const FLOAT* my_mean = means + idx * D;
    const FLOAT my_radius = radii[idx] * 0.333;
    const FLOAT my_inv_radius = 1.0 / my_radius;
    const FLOAT* my_distance_transform = distance_transforms + idx * L * (D*E+1);
    bool* my_indices = indices + idx * P;
    FLOAT* my_dists = dists + idx * D * P;
    FLOAT* my_factors = factors + idx * P * L;
    FLOAT* my_neighbor_feature = neighbor_features + idx * L;

    for (int i = 0; i < P; i++) {
        if (i == idx) continue;
        const FLOAT* other_mean = means + i * D;
        const FLOAT other_radius = radii[i] * 0.333;
        const FLOAT* other_feature = features + i * L;
        FLOAT* other_dists = my_dists + i * D;
        FLOAT* other_factors = my_factors + i * L;

        FLOAT dist = 0.0;
        for (int j = 0; j < D; j++) {
            other_dists[j] = other_mean[j] - my_mean[j];
            dist += other_dists[j] * other_dists[j];
        }

        const FLOAT radius = my_radius + other_radius;
        if (dist < radius * radius) {
            my_indices[i] = true;

            for (int j = 0; j < D; j++) {
                other_dists[j] *= my_inv_radius;
            }
            //     if (other_dists[j] >= 0) {
            //         other_dists[j] = 1.0 / (other_dists[j] / my_radius + 1.0);
            //     } else {
            //         other_dists[j] = -1.0 / (-other_dists[j] / my_radius + 1.0);
            //     }
                // if (other_dists[j] >= 0) {
                //     other_dists[j] = 1.0 - other_dists[j] / my_radius;
                // } else {
                //     other_dists[j] = -(1.0 + other_dists[j] / my_radius);
                // }
            // }

            // TODO: Is there some built-in parallel function to do this for an entire array?
            for (int j = 0; j < L; j++) {
                for (int k = 0; k < D; k++) {
                    if (other_dists[k] >= 0) {
                        other_factors[j] += my_distance_transform[j*(D*E+1) + k*E] / (other_dists[k] + 1.0);
                    } else {
                        other_factors[j] -= my_distance_transform[j*(D*E+1) + k*E] / (-other_dists[k] + 1.0);
                    }
                    for (int l = 0; l < E / 2; l++) {
                        other_factors[j] += my_distance_transform[j*(D*E+1) + k*E + l*2 + 1] * sin(3 * l * M_PI * other_dists[k]);
                        other_factors[j] += my_distance_transform[j*(D*E+1) + k*E + l*2 + 2] * cos(3 * l * M_PI * other_dists[k]);
                    }
                }
                other_factors[j] += my_distance_transform[j*(D*E+1) + D*E];
                my_neighbor_feature[j] += other_factors[j] * other_feature[j];
            }
        }
    }
}

__global__ void findCollisionsBackward(
    const int P, const int D, const int L, const int E,
    const FLOAT* features,
    const bool* indices,
    const FLOAT* dists,
    const FLOAT* factors,
    const FLOAT* inv_counts,
    const FLOAT* dL_dneighbor_features,
    FLOAT* dL_dfeatures,
    FLOAT* dL_ddistance_transforms)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    const bool* my_indices = indices + idx * P;
    const FLOAT* my_factors = factors + idx * P * L;
    const FLOAT* my_dists = dists + idx * D * P;
    const FLOAT my_inv_count = inv_counts[idx];
    const FLOAT* my_dL = dL_dneighbor_features + idx * L;
    FLOAT* my_dL_ddt = dL_ddistance_transforms + idx * L * (D*E+1);

    for (int i = 0; i < P; i++) {
        if (i == idx || !my_indices[i]) continue;
        const FLOAT* other_feature = features + i * L;
        const FLOAT* other_factors = my_factors + i * L;
        const FLOAT* other_dists = my_dists + i * D;
        FLOAT* other_dL_dfeatures = dL_dfeatures + i * L;

        for (int j = 0; j < L; j++) {
            const FLOAT dc = my_dL[j] * my_inv_count;
            atomicAdd(other_dL_dfeatures + j, dc * other_factors[j]);

            FLOAT* other_dL_ddt = my_dL_ddt + j*(D*E+1);

            const FLOAT dcf = dc * other_feature[j];
            for (int k = 0; k < D; k++) {
                if (other_dists[k] >= 0) {
                    other_dL_ddt[k*E] += dcf / (other_dists[k] + 1.0);
                } else {
                    other_dL_ddt[k*E] += -dcf / (-other_dists[k] + 1.0);
                }
                for (int l = 0; l < E / 2; l++) {
                    other_dL_ddt[k*E + l*2 + 1] += dcf * sin(3 * l * M_PI * other_dists[k]);
                    other_dL_ddt[k*E + l*2 + 2] += dcf * cos(3 * l * M_PI * other_dists[k]);
                }
            }
            other_dL_ddt[D*E] += dcf;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> AggregateNeighborsCUDA(
    const torch::Tensor& means,
    const torch::Tensor& radii,
    const torch::Tensor& features,
    const torch::Tensor& distance_transforms,
    const bool debug)
{
    const int P = means.size(0);
    const int D = means.size(-1);
    const int L = features.size(-1);
    const int E = 7;

    torch::Tensor indices = torch::full({P, P}, false, means.options().dtype(torch::kBool));
    torch::Tensor neighbor_features = torch::full({P, L}, 0, means.options());
    torch::Tensor dists = torch::full({P, P, D}, 0, means.options());
    torch::Tensor factors = torch::full({P, P, L}, 1.0, means.options());
    // torch::Tensor bounding_boxes = torch::full({P, D*2}, 0, means.options());

	findCollisions << <(P + 255) / 256, 256 >> > (
		P, D, L, E,
        means.contiguous().data<FLOAT>(),
        radii.contiguous().data<FLOAT>(),
        features.contiguous().data<FLOAT>(),
        distance_transforms.contiguous().data<FLOAT>(),
        indices.contiguous().data<bool>(),
        dists.contiguous().data<FLOAT>(),
        factors.contiguous().data<FLOAT>(),
        neighbor_features.contiguous().data<FLOAT>()
    )
    CHECK_CUDA(, debug)

    const torch::Tensor counts = indices.sum(-1).unsqueeze(-1);
    return std::make_tuple(indices, dists, factors, neighbor_features / counts);
}

std::tuple<torch::Tensor, torch::Tensor> AggregateNeighborsBackwardCUDA(
    const torch::Tensor& features,
    const torch::Tensor& indices,
    const torch::Tensor& dists,
    const torch::Tensor& factors,
    const torch::Tensor& dL_dneighbor_features,
    const bool debug)
{
    const int P = features.size(0);
    const int D = dists.size(-1);
    const int L = features.size(-1);
    const int E = 7;

    const torch::Tensor inv_counts = 1.0 / indices.sum(-1).to(features.dtype());
    torch::Tensor dL_dfeatures = torch::full({P, L}, 0, features.options());
    torch::Tensor dL_ddistance_transforms = torch::full({P, L, D*E+1}, 0, features.options());

	findCollisionsBackward << <(P + 255) / 256, 256 >> > (
		P, D, L, E,
        features.contiguous().data<FLOAT>(),
        indices.contiguous().data<bool>(),
        dists.contiguous().data<FLOAT>(),
        factors.contiguous().data<FLOAT>(),
        inv_counts.contiguous().data<FLOAT>(),
        dL_dneighbor_features.contiguous().data<FLOAT>(),
        dL_dfeatures.contiguous().data<FLOAT>(),
        dL_ddistance_transforms.contiguous().data<FLOAT>()
    )
    CHECK_CUDA(, debug)

    return std::make_tuple(dL_dfeatures, dL_ddistance_transforms);
}
