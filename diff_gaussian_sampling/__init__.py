#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def sample_gaussians(debug, *args):
    return _SampleGaussians.apply(debug, *args)

def sample_gaussians_derivative(debug, *args):
    return _SampleGaussiansDerivative.apply(debug, *args)

def sample_gaussians_laplacian(debug, *args):
    return _SampleGaussiansLaplacian.apply(debug, *args)

def sample_gaussians_third_derivative(debug, *args):
    return _SampleGaussiansThirdDerivative.apply(debug, *args)

def aggregate_neighbors(means, radii, features, distance_transforms, debug):
    return _AggregateNeighbors.apply(means, radii, features, distance_transforms, debug)

def call_debug(func, debug, name, *args):
    if debug:
        cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
        try:
            results = func(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_{}.dump".format(name))
            print("\nAn error occured in {}. Please forward snapshot_{}.dump for debugging.".format(name, name))
            raise ex
    else:
        results = func(*args)

    return results

def preprocess_gaussians(means, values, covariances, conics, samples, debug):
    # Restructure arguments the way that the C++ lib expects them
    args = (
        means,
        values,
        covariances,
        conics,
        samples,
        debug
    )

    results = call_debug(_C.preprocess_gaussians, debug, "preprocess", *args)

    return results

def call_forward(ctx, func, name, *args):
    # Restructure arguments the way that the C++ lib expects them
    means, values, conics, samples, num_rendered, binning_buffer, sample_binning_buffer, ranges, sample_ranges, debug = args

    out_values = call_debug(func, debug, name, *args)

    # Keep relevant tensors for backward
    ctx.debug = debug
    ctx.num_rendered = num_rendered
    ctx.save_for_backward(means, values, conics, samples, binning_buffer, sample_binning_buffer, ranges, sample_ranges)
    return out_values

def call_backward(ctx, func, grad_out, name):
    # Restore necessary values from context
    num_rendered = ctx.num_rendered
    debug = ctx.debug
    means, values, conics, samples, binning_buffer, sample_binning_buffer, ranges, sample_ranges = ctx.saved_tensors

    # Restructure args as C++ method expects them
    args = (
        means,
        values,
        conics,
        samples,
        num_rendered,
        grad_out,
        binning_buffer,
        sample_binning_buffer,
        ranges,
        sample_ranges,
        debug
    )

    grad_means, grad_values, grad_conics = call_debug(func, debug, name, *args)

    return (
        grad_means,
        grad_values,
        grad_conics,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

class _SampleGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        return call_forward(ctx, _C.sample_gaussians, "fw", *args)

    @staticmethod
    def backward(ctx, grad_out):
        return call_backward(ctx, _C.sample_gaussians_backward, grad_out, "bw")

class _SampleGaussiansDerivative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        return call_forward(ctx, _C.sample_gaussians_derivative, "der_fw", *args)

    @staticmethod
    def backward(ctx, grad_out):
        return call_backward(ctx, _C.sample_gaussians_derivative_backward, grad_out, "der_bw")

class _SampleGaussiansLaplacian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        return call_forward(ctx, _C.sample_gaussians_laplacian, "lap_fw", *args)

    @staticmethod
    def backward(ctx, grad_out):
        return call_backward(ctx, _C.sample_gaussians_laplacian_backward, grad_out, "lap_bw")

class _SampleGaussiansThirdDerivative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        return call_forward(ctx, _C.sample_gaussians_third_derivative, "3_fw", *args)

    @staticmethod
    def backward(ctx, grad_out):
        return call_backward(ctx, _C.sample_gaussians_third_derivative_backward, grad_out, "3_bw")


class _AggregateNeighbors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, radii, features, distance_transforms, debug):
        ctx.debug = debug
        ctx.features = features
        ctx.distance_transforms = distance_transforms
        args = (means, radii, features, distance_transforms, debug)
        indices, dists, neighbor_features = call_debug(_C.aggregate_neighbors, debug, "aggregate", *args)
        ctx.indices = indices
        ctx.dists = dists
        return indices, neighbor_features

    @staticmethod
    def backward(ctx, grad_indices, grad_out):
        grad_features, grad_distance_transforms = call_debug(
            _C.aggregate_neighbors_backward, ctx.debug, "aggregate_bw",
            ctx.features, ctx.indices, ctx.dists, ctx.distance_transforms, grad_out, ctx.debug
        )

        return (
            None,
            None,
            grad_features,
            grad_distance_transforms,
            None,
        )

class GaussianSampler:
    def __init__(self, debug):
        self.debug = debug

    def preprocess(self, means, values, covariances, conics, samples):
        debug = self.debug

        num_rendered, binning_buffer, sample_binning_buffer, ranges, sample_ranges, radii = \
            preprocess_gaussians(means, values, covariances, conics, samples, debug)

        self.means = means
        self.values = values
        self.conics = conics
        self.samples = samples
        self.num_rendered = num_rendered
        self.binning_buffer = binning_buffer
        self.sample_binning_buffer = sample_binning_buffer
        self.ranges = ranges
        self.sample_ranges = sample_ranges
        self.radii = radii

    def sample_gaussians(self):
        return sample_gaussians(
            self.means,
            self.values,
            self.conics,
            self.samples,
            self.num_rendered,
            self.binning_buffer,
            self.sample_binning_buffer,
            self.ranges,
            self.sample_ranges,
            self.debug,
        )

    def sample_gaussians_derivative(self):
        return sample_gaussians_derivative(
            self.means,
            self.values,
            self.conics,
            self.samples,
            self.num_rendered,
            self.binning_buffer,
            self.sample_binning_buffer,
            self.ranges,
            self.sample_ranges,
            self.debug,
        )

    def sample_gaussians_laplacian(self):
        return sample_gaussians_laplacian(
            self.means,
            self.values,
            self.conics,
            self.samples,
            self.num_rendered,
            self.binning_buffer,
            self.sample_binning_buffer,
            self.ranges,
            self.sample_ranges,
            self.debug,
        )

    def sample_gaussians_third_derivative(self):
        return sample_gaussians_third_derivative(
            self.means,
            self.values,
            self.conics,
            self.samples,
            self.num_rendered,
            self.binning_buffer,
            self.sample_binning_buffer,
            self.ranges,
            self.sample_ranges,
            self.debug,
        )

    def aggregate_neighbors(self, features, distance_transforms):
        return aggregate_neighbors(self.means, self.radii, features, distance_transforms, self.debug)
