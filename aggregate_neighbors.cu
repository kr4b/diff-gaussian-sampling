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
    const FLOAT* conics,
    const FLOAT* radii,
    const FLOAT* features,
    const FLOAT* frequencies,
    const FLOAT* distance_transforms,
    bool* indices,
    FLOAT* dists,
    FLOAT* inv_total_densities,
    FLOAT* neighbor_features)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    const FLOAT my_radius = radii[idx] * 0.333;
    if (my_radius < 1e-6) return;

    const FLOAT* my_mean = means + idx * D;
    const FLOAT my_inv_radius = 1.0 / (my_radius + 1e-6);
    const FLOAT* my_distance_transform = distance_transforms + idx * L * E;
    bool* my_indices = indices + idx * P;
    FLOAT* my_dists = dists + idx * D * P;
    FLOAT* my_inv_total_density = inv_total_densities + idx;
    FLOAT* my_neighbor_feature = neighbor_features + idx * L;

    FLOAT total_density = 0.0;
    for (int i = 0; i < P; i++) {
        const FLOAT* other_mean = means + i * D;
        const FLOAT* con = conics + i * (D * (D+1) / 2);
        const FLOAT other_radius = radii[i] * 0.333;
        const FLOAT* other_feature = features + i * L;
        FLOAT* X = my_dists + i * D;

        if (other_radius < 1e-6) continue;

        FLOAT dist = 0.0;
        for (int d = 0; d < D; d++) {
            X[d] = other_mean[d] - my_mean[d];
            dist += X[d] * X[d];
        }

        const FLOAT radius = my_radius + other_radius;
        if (dist > radius * radius) continue;

        FLOAT power = 0.0;
        if (D == 1) {
            power = -0.5 * con[0] * X[0] * X[0];
        } else if (D == 2) {
            power = -0.5 * (con[0] * X[0] * X[0] + con[2] * X[1] * X[1]) - con[1] * X[0] * X[1];
        }

        if (power > 0) continue;
        const FLOAT density = exp(power);
        total_density += density;

        my_indices[i] = true;

        for (int j = 0; j < L; j++) {
            FLOAT factor = 0.0;
            for (int d = 0; d < D; d++) {
                for (int e = 0; e < (E-D-1)/D/2; e++) {
                    factor += my_distance_transform[j*E + d*((E-1)/D) + e*2 + 1] * sin(frequencies[e] * M_PI * X[d] * my_inv_radius);
                    factor += my_distance_transform[j*E + d*((E-1)/D) + e*2 + 2] * cos(frequencies[e] * M_PI * X[d] * my_inv_radius);
                }
            }
            factor += my_distance_transform[j*E + E - 1];
            my_neighbor_feature[j] += density * factor * other_feature[j];
        }
    }

    const FLOAT inv_total_density = 1.0 / (total_density + 1e-6);
    *my_inv_total_density = inv_total_density;
    for (int j = 0; j < L; j++) {
        my_neighbor_feature[j] *= inv_total_density;
    }
}

__global__ void findCollisionsBackward(
    const int P, const int D, const int L, const int E,
    const FLOAT* conics,
    const FLOAT* radii,
    const FLOAT* features,
    const bool* indices,
    const FLOAT* dists,
    const FLOAT* inv_total_densities,
    const FLOAT* frequencies,
    const FLOAT* distance_transforms,
    const FLOAT* dL_dneighbor_features,
    FLOAT* dL_dfeatures,
    FLOAT* dL_dfrequencies,
    FLOAT* dL_ddistance_transforms)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    const FLOAT my_radius = radii[idx] * 0.333;
    if (my_radius < 1e-6) return;

    const FLOAT my_inv_radius = 1.0 / (my_radius + 1e-6);
    const bool* my_indices = indices + idx * P;
    const FLOAT* my_dists = dists + idx * D * P;
    const FLOAT my_inv_total_density = inv_total_densities[idx];
    const FLOAT* my_distance_transform = distance_transforms + idx * L * E;
    const FLOAT* my_dL = dL_dneighbor_features + idx * L;
    FLOAT* my_dL_ddt = dL_ddistance_transforms + idx * L * E;

    for (int i = 0; i < P; i++) {
        if (!my_indices[i]) continue;

        const FLOAT* con = conics + i * (D * (D+1) / 2);
        const FLOAT* X = my_dists + i * D;
        const FLOAT* other_feature = features + i * L;
        FLOAT* other_dL_dfeatures = dL_dfeatures + i * L;

        FLOAT power = 0.0;
        if (D == 1) {
            power = -0.5 * con[0] * X[0] * X[0];
        } else if (D == 2) {
            power = -0.5 * (con[0] * X[0] * X[0] + con[2] * X[1] * X[1]) - con[1] * X[0] * X[1];
        }

        if (power > 0) continue;
        const FLOAT density = exp(power);

        for (int j = 0; j < L; j++) {
            const FLOAT dc = my_dL[j] * density * my_inv_total_density;
            const FLOAT dcf = dc * other_feature[j];

            FLOAT factor = 0.0;
            for (int d = 0; d < D; d++) {
                for (int e = 0; e < (E-D-1)/D/2; e++) {
                    const FLOAT s = sin(frequencies[e] * M_PI * X[d] * my_inv_radius);
                    const FLOAT c = cos(frequencies[e] * M_PI * X[d] * my_inv_radius);

                    factor += my_distance_transform[j*E + d*((E-1)/D) + e*2 + 1] * s;
                    my_dL_ddt[j*E + d*((E-1)/D) + e*2 + 1] += dcf * s;
                    atomicAdd(dL_dfrequencies + e, my_distance_transform[j*E + d*((E-1)/D) + e*2 + 1] * c * M_PI * X[d] * my_inv_radius * dcf);

                    factor += my_distance_transform[j*E + d*((E-1)/D) + e*2 + 2] * c;
                    my_dL_ddt[j*E + d*((E-1)/D) + e*2 + 2] += dcf * c;
                    atomicAdd(dL_dfrequencies + e, -my_distance_transform[j*E + d*((E-1)/D) + e*2 + 2] * s * M_PI * X[d] * my_inv_radius * dcf);
                }
            }
            my_dL_ddt[j*E + E - 1] += dcf;
            factor += my_distance_transform[j*E + E - 1];

            atomicAdd(other_dL_dfeatures + j, dc * factor);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> AggregateNeighborsCUDA(
    const torch::Tensor& means,
    const torch::Tensor& conics,
    const torch::Tensor& radii,
    const torch::Tensor& features,
    const torch::Tensor& frequencies,
    const torch::Tensor& distance_transforms,
    const bool debug)
{
    const int P = means.size(0);
    const int D = means.size(-1);
    const int L = features.size(-1);
    const int E = distance_transforms.size(-1);

    torch::Tensor indices = torch::full({P, P}, false, means.options().dtype(torch::kBool));
    torch::Tensor neighbor_features = torch::full({P, L}, 0, means.options());
    torch::Tensor dists = torch::full({P, P, D}, 0, means.options());
    torch::Tensor inv_total_densities = torch::full({P}, 0, means.options());
    // torch::Tensor bounding_boxes = torch::full({P, D*2}, 0, means.options());

	findCollisions << <(P + 255) / 256, 256 >> > (
		P, D, L, E,
        means.contiguous().data<FLOAT>(),
        conics.contiguous().data<FLOAT>(),
        radii.contiguous().data<FLOAT>(),
        features.contiguous().data<FLOAT>(),
        frequencies.contiguous().data<FLOAT>(),
        distance_transforms.contiguous().data<FLOAT>(),
        indices.contiguous().data<bool>(),
        dists.contiguous().data<FLOAT>(),
        inv_total_densities.contiguous().data<FLOAT>(),
        neighbor_features.contiguous().data<FLOAT>()
    )
    CHECK_CUDA(, debug)

    return std::make_tuple(indices, dists, inv_total_densities, neighbor_features);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AggregateNeighborsBackwardCUDA(
    const torch::Tensor& conics,
    const torch::Tensor& radii,
    const torch::Tensor& features,
    const torch::Tensor& indices,
    const torch::Tensor& dists,
    const torch::Tensor& inv_total_densities,
    const torch::Tensor& frequencies,
    const torch::Tensor& distance_transforms,
    const torch::Tensor& dL_dneighbor_features,
    const bool debug)
{
    const int P = features.size(0);
    const int D = dists.size(-1);
    const int L = features.size(-1);
    const int E = distance_transforms.size(-1);

    torch::Tensor dL_dfeatures = torch::full(features.sizes(), 0, features.options());
    torch::Tensor dL_dfrequencies = torch::full(frequencies.sizes(), 0, features.options());
    torch::Tensor dL_ddistance_transforms = torch::full(distance_transforms.sizes(), 0, features.options());

	findCollisionsBackward << <(P + 255) / 256, 256 >> > (
		P, D, L, E,
        conics.contiguous().data<FLOAT>(),
        radii.contiguous().data<FLOAT>(),
        features.contiguous().data<FLOAT>(),
        indices.contiguous().data<bool>(),
        dists.contiguous().data<FLOAT>(),
        inv_total_densities.contiguous().data<FLOAT>(),
        frequencies.contiguous().data<FLOAT>(),
        distance_transforms.contiguous().data<FLOAT>(),
        dL_dneighbor_features.contiguous().data<FLOAT>(),
        dL_dfeatures.contiguous().data<FLOAT>(),
        dL_dfrequencies.contiguous().data<FLOAT>(),
        dL_ddistance_transforms.contiguous().data<FLOAT>()
    )
    CHECK_CUDA(, debug)

    return std::make_tuple(dL_dfeatures, dL_dfrequencies, dL_ddistance_transforms);
}
