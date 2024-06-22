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
    const int P, const int D,
    const FLOAT* means,
    const FLOAT* radii,
    bool* indices)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    const FLOAT my_radius = radii[idx] * 0.2;
    if (my_radius < 1e-6) return;

    const FLOAT* my_mean = means + idx * D;
    const FLOAT my_inv_radius = 1.0 / (my_radius + 1e-6);
    bool* my_indices = indices + idx * P;

    for (int i = 0; i < P; i++) {
        // if (idx == i) continue;
        const FLOAT* other_mean = means + i * D;
        const FLOAT other_radius = radii[i] * 0.2;

        if (other_radius < 1e-6) continue;

        FLOAT dist = 0.0;
        for (int d = 0; d < D; d++) {
            FLOAT dx = other_mean[d] - my_mean[d];
#ifdef TORUS
            dx = min(dx, abs(2.0 - fmod(abs(dx), 2.0)));
#endif
            dist += dx * dx;
        }

        const FLOAT radius = my_radius + other_radius;
        if (dist > radius * radius) continue;
        my_indices[i] = true;
    }
}

__global__ void preprocess(
    const int P, const int D,
    const FLOAT* means,
    const FLOAT* conics,
    const FLOAT* radii,
    const bool* bool_indices,
    const long* ranges,
    long* indices,
    FLOAT* dists,
    FLOAT* densities,
    FLOAT* inv_total_densities)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    const FLOAT* my_mean = means + idx * D;
    const FLOAT my_radius = radii[idx] * 0.333;
    const FLOAT my_inv_radius = 1.0 / (my_radius + 1e-6);
    const bool* my_bool_indices = bool_indices + idx * P;
    const long start = (idx == 0) ? 0 : ranges[idx - 1];
    long* my_indices = indices + start;
    FLOAT* my_dists = dists + start * D;
    FLOAT* my_densities = densities + start;
    FLOAT* my_inv_total_density = inv_total_densities + idx;

    FLOAT total_density = 0.0;
    int current = -1;
    for (int i = 0; i < P; i++) {
        if (!my_bool_indices[i]) continue;
        current += 1;
        const FLOAT* other_mean = means + i * D;
        const FLOAT* con = conics + i * (D * (D+1) / 2);
        FLOAT* X = my_dists + current * D;

        for (int d = 0; d < D; d++) {
            X[d] = other_mean[d] - my_mean[d];
#ifdef TORUS
            if (abs(X[d]) > 1.0) {
                if (X[d] >= 0) {
                    X[d] = fmod(X[d], 2.0) - 2.0;
                } else {
                    X[d] = fmod(X[d], 2.0) + 2.0;
                }
            } else {
                X[d] = X[d];
            }
#endif
        }

        FLOAT power = 0.0;
        if (D == 1) {
            power = -0.5 * con[0] * X[0] * X[0];
        } else if (D == 2) {
            power = -0.5 * (con[0] * X[0] * X[0] + con[2] * X[1] * X[1]) - con[1] * X[0] * X[1];
        }

        for (int d = 0; d < D; d++) {
            X[d] *= my_inv_radius;
        }

        if (power > 0) continue;

        my_densities[current] = exp(power);
        my_indices[current] = i;
        total_density += my_densities[current];
    }

    const FLOAT inv_total_density = 1.0 / (total_density + 1e-6);
    *my_inv_total_density = inv_total_density;
}

__global__ void aggregateNeighbors(
    const int P, const int D, const int L, const int K, const int E,
    const FLOAT* features,
    const FLOAT* transform,
    const FLOAT* queries,
    const FLOAT* keys,
    const FLOAT* frequencies,
    const FLOAT* distance_transform,
    const long* indices,
    const long* ranges,
    const FLOAT* dists,
    const FLOAT* densities,
    const FLOAT* inv_total_densities,
    FLOAT* weights,
    FLOAT* embeddings,
    FLOAT* factors,
    FLOAT* neighbor_features)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    const FLOAT* my_query = queries + idx * K;
    const long my_range = ranges[idx];
    const long start = (idx == 0) ? 0 : ranges[idx - 1];
    const long* my_indices = indices + start;
    const FLOAT* my_dists = dists + start * D;
    const FLOAT* my_densities = densities + start;
    const FLOAT my_inv_total_density = inv_total_densities[idx];
    FLOAT* my_weights = weights + start;
    FLOAT* my_embeddings = embeddings + start;
    FLOAT* my_factors = factors + start;
    FLOAT* my_neighbor_feature = neighbor_features + idx * L;

    FLOAT total_density = 0.0;
    for (long i = 0; i < my_range - start; i++) {
        const long index = my_indices[i];
        if (index == -1) continue;
        const FLOAT* other_feature = features + index * L;
        const FLOAT* other_key = keys + index * K;
        const FLOAT* X = my_dists + i * D;
        const FLOAT density = my_densities[i];
        FLOAT weight = 0.0;

        for (int k = 0; k < K; k++) {
            weight += my_query[k] * other_key[k];
        }
        my_weights[i] = weight;

        FLOAT embedding = 0.0;
        FLOAT factor = 0.0;
        for (int d = 0; d < D; d++) {
            for (int e = 0; e < (E-1)/D/2; e++) {
                const FLOAT s = sin(frequencies[e] * M_PI * X[d]);
                const FLOAT c = cos(frequencies[e] * M_PI * X[d]);

                embedding += distance_transform[d*((E-1)/D) + e*2 + 0] * s;
                embedding += distance_transform[d*((E-1)/D) + e*2 + 1] * c;
                factor += distance_transform[E + d*((E-1)/D) + e*2 + 0] * s;
                factor += distance_transform[E + d*((E-1)/D) + e*2 + 1] * c;
            }
        }

        embedding += distance_transform[E - 1];
        factor += distance_transform[2*E - 1];

        my_embeddings[i] = embedding;
        my_factors[i] = factor;

        const FLOAT dw = my_inv_total_density * density * weight;
        const FLOAT dwf = dw * factor;
        const FLOAT dwe = dw * embedding;
        for (int j = 0; j < L; j++) {
            const FLOAT embedded = dwe + dwf * other_feature[j];
            for (int k = 0; k < L; k++) {
                my_neighbor_feature[k] += transform[j * L + k] * embedded;
            }
        }
    }
}

__global__ void aggregateNeighborsBackward(
    const int P, const int D, const int L, const int K, const int E,
    const FLOAT* features,
    const FLOAT* transform,
    const FLOAT* queries,
    const FLOAT* keys,
    const FLOAT* frequencies,
    const FLOAT* distance_transform,
    const long* indices,
    const long* ranges,
    const FLOAT* dists,
    const FLOAT* densities,
    const FLOAT* weights,
    const FLOAT* embeddings,
    const FLOAT* factors,
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

    const FLOAT* my_query = queries + idx * K;
    const long my_range = ranges[idx];
    const long start = (idx == 0) ? 0 : ranges[idx - 1];
    const long* my_indices = indices + start;
    const FLOAT* my_dists = dists + start * D;
    const FLOAT* my_densities = densities + start;
    const FLOAT* my_weights = weights + start;
    const FLOAT* my_embeddings = embeddings + start;
    const FLOAT* my_factors = factors + start;
    const FLOAT my_inv_total_density = inv_total_densities[idx];
    const FLOAT* my_dL = dL_dneighbor_features + idx * L;
    FLOAT* my_dL_dqueries = dL_dqueries + idx * K;

    FLOAT* summed_transform = new FLOAT[L];

    for (int j = 0; j < L; j++) {
        summed_transform[j] = 0.0;
        for (int k = 0; k < L; k++) {
            summed_transform[j] += transform[j * L + k] * my_dL[k];
        }
    }

    for (long i = 0; i < my_range - start; i++) {
        const long index = my_indices[i];
        if (index == -1) continue;
        const FLOAT* other_feature = features + index * L;
        const FLOAT* other_key = keys + index * K;
        const FLOAT* X = my_dists + i * D;
        const FLOAT density = my_densities[i];
        const FLOAT weight = my_weights[i];
        const FLOAT embedding = my_embeddings[i];
        const FLOAT factor = my_factors[i];
        FLOAT* other_dL_dfeatures = dL_dfeatures + index * L;
        FLOAT* other_dL_dkeys = dL_dkeys + index * K;

        const FLOAT dc = density * my_inv_total_density;
        const FLOAT dcw = dc * weight;

        for (int d = 0; d < D; d++) {
            for (int e = 0; e < (E-1)/D/2; e++) {
                const FLOAT s = sin(frequencies[e] * M_PI * X[d]);
                const FLOAT c = cos(frequencies[e] * M_PI * X[d]);

                for (int j = 0; j < L; j++) {
                    const FLOAT dct = dcw * summed_transform[j];

                    atomicAdd(dL_ddt + d*((E-1)/D) + e*2 + 0, dct * s);
                    atomicAdd(dL_ddt + E + d*((E-1)/D) + e*2 + 0, dct * s * other_feature[j]);
                    atomicAdd(dL_dfrequencies + e, c * M_PI * X[d] * dct * (distance_transform[d*((E-1)/D) + e*2 + 0] + distance_transform[E + d*((E-1)/D) + e*2 + 0] * other_feature[j]));

                    atomicAdd(dL_ddt + d*((E-1)/D) + e*2 + 1, dct * c);
                    atomicAdd(dL_ddt + E + d*((E-1)/D) + e*2 + 1, dct * c * other_feature[j]);
                    atomicAdd(dL_dfrequencies + e, -s * M_PI * X[d] * dct * (distance_transform[d*((E-1)/D) + e*2 + 1] + distance_transform[E + d*((E-1)/D) + e*2 + 1] * other_feature[j]));
                }
            }
        }

        const FLOAT dce = dc * embedding;
        const FLOAT dcf = dc * factor;

        for (int j = 0; j < L; j++) {
            const FLOAT dct = dcw * summed_transform[j];

            atomicAdd(dL_ddt + E - 1, dct);
            atomicAdd(dL_ddt + 2*E - 1, dct * other_feature[j]);
            atomicAdd(other_dL_dfeatures + j, dct * factor);

            const FLOAT embedded = dce + dcf * other_feature[j];

            const FLOAT we = weight * embedded;
            for (int k = 0; k < L; k++) {
                atomicAdd(dL_dtransform + j * L + k, we * my_dL[k]);
            }

            const FLOAT te = summed_transform[j] * embedded;
            for (int k = 0; k < K; k++) {
                atomicAdd(my_dL_dqueries + k, other_key[k] * te);
                atomicAdd(other_dL_dkeys + k, my_query[k] * te);
            }
        }
    }

    delete summed_transform;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> AggregateNeighborsPreprocessCUDA(
    const torch::Tensor& means,
    const torch::Tensor& conics,
    const torch::Tensor& radii,
    const bool debug)
{
    const int P = means.size(0);
    const int D = means.size(-1);

    torch::Tensor bool_indices = torch::full({P, P}, false, means.options().dtype(torch::kBool));
    // torch::Tensor bounding_boxes = torch::full({P, D*2}, 0, means.options());

	findCollisions <<< (P + 255) / 256, 256 >>> (
		P, D,
        means.contiguous().data<FLOAT>(),
        radii.contiguous().data<FLOAT>(),
        bool_indices.contiguous().data<bool>()
    )
    CHECK_CUDA(, debug)

    const torch::Tensor counts = bool_indices.sum(-1);
    const torch::Tensor ranges = counts.cumsum(0);
    const int length = ranges[-1].item<int>();

    torch::Tensor indices = torch::full({length}, -1, means.options().dtype(torch::kLong));
    torch::Tensor dists = torch::full({length, D}, 0, means.options());
    torch::Tensor densities = torch::full({length}, 0, means.options());
    torch::Tensor inv_total_densities = torch::full({P}, 0, means.options());

    preprocess <<< (P + 255) / 256, 256 >>> (
        P, D,
        means.contiguous().data<FLOAT>(),
        conics.contiguous().data<FLOAT>(),
        radii.contiguous().data<FLOAT>(),
        bool_indices.contiguous().data<bool>(),
        ranges.contiguous().data<long>(),
        indices.contiguous().data<long>(),
        dists.contiguous().data<FLOAT>(),
        densities.contiguous().data<FLOAT>(),
        inv_total_densities.contiguous().data<FLOAT>()
    )
    CHECK_CUDA(, debug)

    return std::make_tuple(indices, ranges, dists, densities, inv_total_densities);
}

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
    const bool debug)
{
    const int P = features.size(0);
    const int D = dists.size(-1);
    const int L = features.size(-1);
    const int K = queries.size(-1);
    const int E = distance_transform.size(-1)/2;

    torch::Tensor weights = torch::full(densities.sizes(), 0, features.options());
    torch::Tensor embeddings = torch::full(densities.sizes(), 0, features.options());
    torch::Tensor factors = torch::full(densities.sizes(), 0, features.options());
    torch::Tensor neighbor_features = torch::full({P, L}, 0, features.options());

	aggregateNeighbors <<< (P + 255) / 256, 256 >>> (
		P, D, L, K, E,
        features.contiguous().data<FLOAT>(),
        transform.contiguous().data<FLOAT>(),
        queries.contiguous().data<FLOAT>(),
        keys.contiguous().data<FLOAT>(),
        frequencies.contiguous().data<FLOAT>(),
        distance_transform.contiguous().data<FLOAT>(),
        indices.contiguous().data<long>(),
        ranges.contiguous().data<long>(),
        dists.contiguous().data<FLOAT>(),
        densities.contiguous().data<FLOAT>(),
        inv_total_densities.contiguous().data<FLOAT>(),
        weights.contiguous().data<FLOAT>(),
        embeddings.contiguous().data<FLOAT>(),
        factors.contiguous().data<FLOAT>(),
        neighbor_features.contiguous().data<FLOAT>()
    )
    CHECK_CUDA(, debug)

    return std::make_tuple(weights, embeddings, factors, neighbor_features);
}

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

	aggregateNeighborsBackward <<< (P + 255) / 256, 256 >>> (
		P, D, L, K, E,
        features.contiguous().data<FLOAT>(),
        transform.contiguous().data<FLOAT>(),
        queries.contiguous().data<FLOAT>(),
        keys.contiguous().data<FLOAT>(),
        frequencies.contiguous().data<FLOAT>(),
        distance_transform.contiguous().data<FLOAT>(),
        indices.contiguous().data<long>(),
        ranges.contiguous().data<long>(),
        dists.contiguous().data<FLOAT>(),
        densities.contiguous().data<FLOAT>(),
        weights.contiguous().data<FLOAT>(),
        embeddings.contiguous().data<FLOAT>(),
        factors.contiguous().data<FLOAT>(),
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
