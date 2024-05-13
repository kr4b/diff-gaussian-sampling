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
    const int P, const int D, const int L, const int K, const int E,
    const FLOAT* means,
    const FLOAT* conics,
    const FLOAT* radii,
    const FLOAT* features,
    const FLOAT* transform,
    const FLOAT* queries,
    const FLOAT* keys,
    const FLOAT* frequencies,
    const FLOAT* distance_transform,
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
    const FLOAT* my_query = queries + idx * K;
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
        const FLOAT* other_key = keys + i * K;
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

        FLOAT weight = 0.0;
        for (int k = 0; k < K; k++) {
            weight += my_query[k] * other_key[k];
        }

        for (int j = 0; j < L; j++) {
            FLOAT embedding = 0.0;
            FLOAT factor = 0.0;
            for (int d = 0; d < D; d++) {
                for (int e = 0; e < (E-1)/D/2; e++) {
                    embedding += distance_transform[d*((E-1)/D) + e*2 + 0] * sin(frequencies[e] * M_PI * X[d] * my_inv_radius);
                    embedding += distance_transform[d*((E-1)/D) + e*2 + 1] * cos(frequencies[e] * M_PI * X[d] * my_inv_radius);
                    factor += distance_transform[E + d*((E-1)/D) + e*2 + 0] * sin(frequencies[e] * M_PI * X[d] * my_inv_radius);
                    factor += distance_transform[E + d*((E-1)/D) + e*2 + 1] * cos(frequencies[e] * M_PI * X[d] * my_inv_radius);
                }
            }

            embedding += distance_transform[E - 1];
            factor += distance_transform[2*E - 1];

            const FLOAT embedded = density * weight * (embedding + factor * other_feature[j]);
            for (int k = 0; k < L; k++) {
                my_neighbor_feature[k] += transform[j * L + k] * embedded;
            }
        }
    }

    const FLOAT inv_total_density = 1.0 / (total_density + 1e-6);
    *my_inv_total_density = inv_total_density;
    for (int j = 0; j < L; j++) {
        my_neighbor_feature[j] *= inv_total_density;
    }
}

__global__ void findCollisionsBackward(
    const int P, const int D, const int L, const int K, const int E,
    const FLOAT* conics,
    const FLOAT* radii,
    const FLOAT* features,
    const FLOAT* transform,
    const FLOAT* queries,
    const FLOAT* keys,
    const FLOAT* frequencies,
    const FLOAT* distance_transform,
    const bool* indices,
    const FLOAT* dists,
    const FLOAT* inv_total_densities,
    const FLOAT* dL_dneighbor_features,
    FLOAT* dL_dfeatures,
    FLOAT* dL_dtransform,
    FLOAT* dL_dqueries,
    FLOAT* dL_dkeys,
    FLOAT* dL_dfrequencies,
    FLOAT* dL_ddt)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    const FLOAT my_radius = radii[idx] * 0.333;
    if (my_radius < 1e-6) return;

    const FLOAT my_inv_radius = 1.0 / (my_radius + 1e-6);
    const FLOAT* my_query = queries + idx * K;
    const bool* my_indices = indices + idx * P;
    const FLOAT* my_dists = dists + idx * D * P;
    const FLOAT my_inv_total_density = inv_total_densities[idx];
    const FLOAT* my_dL = dL_dneighbor_features + idx * L;
    FLOAT* my_dL_dqueries = dL_dqueries + idx * K;

    FLOAT* summed_transform = new FLOAT[L];

    for (int j = 0; j < L; j++) {
        summed_transform[j] = 0.0;
    }

    for (int j = 0; j < L; j++) {
        for (int k = 0; k < L; k++) {
            summed_transform[j] += transform[j * L + k] * my_dL[k];
        }
    }

    for (int i = 0; i < P; i++) {
        if (!my_indices[i]) continue;

        const FLOAT* con = conics + i * (D * (D+1) / 2);
        const FLOAT* X = my_dists + i * D;
        const FLOAT* other_feature = features + i * L;
        const FLOAT* other_key = keys + i * K;
        FLOAT* other_dL_dfeatures = dL_dfeatures + i * L;
        FLOAT* other_dL_dkeys = dL_dkeys + i * K;

        FLOAT power = 0.0;
        if (D == 1) {
            power = -0.5 * con[0] * X[0] * X[0];
        } else if (D == 2) {
            power = -0.5 * (con[0] * X[0] * X[0] + con[2] * X[1] * X[1]) - con[1] * X[0] * X[1];
        }

        if (power > 0) continue;
        const FLOAT density = exp(power);

        FLOAT weight = 0.0;
        for (int k = 0; k < K; k++) {
            weight += my_query[k] * other_key[k];
        }

        for (int j = 0; j < L; j++) {
            const FLOAT dc = density * my_inv_total_density;
            const FLOAT dcf = dc * weight * summed_transform[j];

            FLOAT embedding = 0.0;
            FLOAT factor = 0.0;
            for (int d = 0; d < D; d++) {
                for (int e = 0; e < (E-1)/D/2; e++) {
                    const FLOAT s = sin(frequencies[e] * M_PI * X[d] * my_inv_radius);
                    const FLOAT c = cos(frequencies[e] * M_PI * X[d] * my_inv_radius);

                    embedding += distance_transform[d*((E-1)/D) + e*2 + 0] * s;
                    factor += distance_transform[E + d*((E-1)/D) + e*2 + 0] * s;
                    atomicAdd(dL_ddt + d*((E-1)/D) + e*2 + 0, dcf * s);
                    atomicAdd(dL_ddt + E + d*((E-1)/D) + e*2 + 0, dcf * s * other_feature[j]);
                    atomicAdd(dL_dfrequencies + e, c * M_PI * X[d] * my_inv_radius * dcf * (distance_transform[d*((E-1)/D) + e*2 + 0] + distance_transform[E + d*((E-1)/D) + e*2 + 0] * other_feature[j]));

                    embedding += distance_transform[d*((E-1)/D) + e*2 + 1] * c;
                    factor += distance_transform[E + d*((E-1)/D) + e*2 + 1] * c;
                    atomicAdd(dL_ddt + d*((E-1)/D) + e*2 + 1, dcf * c);
                    atomicAdd(dL_ddt + E + d*((E-1)/D) + e*2 + 1, dcf * c * other_feature[j]);
                    atomicAdd(dL_dfrequencies + e, -s * M_PI * X[d] * my_inv_radius * dcf * (distance_transform[d*((E-1)/D) + e*2 + 1] + distance_transform[E + d*((E-1)/D) + e*2 + 1] * other_feature[j]));
                }
            }

            embedding += distance_transform[E - 1];
            factor += distance_transform[E + E - 1];
            atomicAdd(dL_ddt + E - 1, dcf);
            atomicAdd(dL_ddt + 2*E - 1, dcf * other_feature[j]);
            atomicAdd(other_dL_dfeatures + j, dcf * factor);

            const FLOAT embedded = dc * (embedding + factor * other_feature[j]);
            for (int k = 0; k < L; k++) {
                atomicAdd(dL_dtransform + j * L + k, weight * embedded * my_dL[k]);
            }
            for (int k = 0; k < K; k++) {
                atomicAdd(my_dL_dqueries + k, other_key[k] * summed_transform[j] * embedded);
                atomicAdd(other_dL_dkeys + k, my_query[k] * summed_transform[j] * embedded);
            }
        }
    }

    delete summed_transform;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> AggregateNeighborsCUDA(
    const torch::Tensor& means,
    const torch::Tensor& conics,
    const torch::Tensor& radii,
    const torch::Tensor& features,
    const torch::Tensor& transform,
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& frequencies,
    const torch::Tensor& distance_transform,
    const bool debug)
{
    const int P = means.size(0);
    const int D = means.size(-1);
    const int L = features.size(-1);
    const int K = queries.size(-1);
    const int E = distance_transform.size(-1)/2;

    torch::Tensor indices = torch::full({P, P}, false, means.options().dtype(torch::kBool));
    torch::Tensor neighbor_features = torch::full({P, L}, 0, means.options());
    torch::Tensor dists = torch::full({P, P, D}, 0, means.options());
    torch::Tensor inv_total_densities = torch::full({P}, 0, means.options());
    // torch::Tensor bounding_boxes = torch::full({P, D*2}, 0, means.options());

	findCollisions << <(P + 255) / 256, 256 >> > (
		P, D, L, K, E,
        means.contiguous().data<FLOAT>(),
        conics.contiguous().data<FLOAT>(),
        radii.contiguous().data<FLOAT>(),
        features.contiguous().data<FLOAT>(),
        transform.contiguous().data<FLOAT>(),
        queries.contiguous().data<FLOAT>(),
        keys.contiguous().data<FLOAT>(),
        frequencies.contiguous().data<FLOAT>(),
        distance_transform.contiguous().data<FLOAT>(),
        indices.contiguous().data<bool>(),
        dists.contiguous().data<FLOAT>(),
        inv_total_densities.contiguous().data<FLOAT>(),
        neighbor_features.contiguous().data<FLOAT>()
    )
    CHECK_CUDA(, debug)

    return std::make_tuple(indices, dists, inv_total_densities, neighbor_features);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> AggregateNeighborsBackwardCUDA(
    const torch::Tensor& conics,
    const torch::Tensor& radii,
    const torch::Tensor& features,
    const torch::Tensor& transform,
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& indices,
    const torch::Tensor& dists,
    const torch::Tensor& inv_total_densities,
    const torch::Tensor& frequencies,
    const torch::Tensor& distance_transform,
    const torch::Tensor& dL_dneighbor_features,
    const bool debug)
{
    const int P = features.size(0);
    const int D = dists.size(-1);
    const int L = features.size(-1);
    const int K = queries.size(-1);
    const int E = distance_transform.size(-1)/2;

    torch::Tensor dL_dfeatures = torch::full(features.sizes(), 0, features.options());
    torch::Tensor dL_dtransform = torch::full(transform.sizes(), 0, features.options());
    torch::Tensor dL_dqueries = torch::full(queries.sizes(), 0, features.options());
    torch::Tensor dL_dkeys = torch::full(keys.sizes(), 0, features.options());
    torch::Tensor dL_dfrequencies = torch::full(frequencies.sizes(), 0, features.options());
    torch::Tensor dL_ddistance_transform = torch::full(distance_transform.sizes(), 0, features.options());

	findCollisionsBackward << <(P + 255) / 256, 256 >> > (
		P, D, L, K, E,
        conics.contiguous().data<FLOAT>(),
        radii.contiguous().data<FLOAT>(),
        features.contiguous().data<FLOAT>(),
        transform.contiguous().data<FLOAT>(),
        queries.contiguous().data<FLOAT>(),
        keys.contiguous().data<FLOAT>(),
        frequencies.contiguous().data<FLOAT>(),
        distance_transform.contiguous().data<FLOAT>(),
        indices.contiguous().data<bool>(),
        dists.contiguous().data<FLOAT>(),
        inv_total_densities.contiguous().data<FLOAT>(),
        dL_dneighbor_features.contiguous().data<FLOAT>(),
        dL_dfeatures.contiguous().data<FLOAT>(),
        dL_dtransform.contiguous().data<FLOAT>(),
        dL_dqueries.contiguous().data<FLOAT>(),
        dL_dkeys.contiguous().data<FLOAT>(),
        dL_dfrequencies.contiguous().data<FLOAT>(),
        dL_ddistance_transform.contiguous().data<FLOAT>()
    )
    CHECK_CUDA(, debug)

    return std::make_tuple(dL_dfeatures, dL_dtransform, dL_dqueries, dL_dkeys, dL_dfrequencies, dL_ddistance_transform);
}
