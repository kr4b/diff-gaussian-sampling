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

def call_debug(func, debug, name, *args):
    if debug:
        cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
        try:
            results = func(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_preprocess.dump")
            print("\nAn error occured in preprocessing. Please forward snapshot_preprocess.dump for debugging.")
            raise ex
    else:
        results = func(*args)

    return results

def preprocess_gaussians(
    means,
    values,
    covariances,
    conics,
    opacities,
    samples,
    debug,
):
    # Restructure arguments the way that the C++ lib expects them
    args = (
        means,
        values,
        covariances,
        conics,
        opacities,
        samples,
        debug
    )

    results = call_debug(_C.preprocess_gaussians, debug, "preprocess", *args)

    return results

def call_forward(ctx, func, name, *args):
    # Restructure arguments the way that the C++ lib expects them
    means, values, conics, opacities, samples, num_rendered, binning_buffer, sample_binning_buffer, ranges, sample_ranges, debug = args

    out_values = call_debug(func, debug, name, *args)

    # Keep relevant tensors for backward
    ctx.debug = debug
    ctx.num_rendered = num_rendered
    ctx.save_for_backward(means, values, conics, opacities, samples, binning_buffer, sample_binning_buffer, ranges, sample_ranges)
    return out_values

def call_backward(ctx, func, grad_out, name):
    # Restore necessary values from context
    num_rendered = ctx.num_rendered
    debug = ctx.debug
    means, values, conics, opacities, samples, binning_buffer, sample_binning_buffer, ranges, sample_ranges = ctx.saved_tensors

    # Restructure args as C++ method expects them
    args = (
        means,
        values,
        conics,
        opacities,
        samples,
        num_rendered,
        grad_out,
        binning_buffer,
        sample_binning_buffer,
        ranges,
        sample_ranges,
        debug
    )

    grad_means, grad_values, grad_conics, grad_opacities, grad_samples = call_debug(func, debug, name, *args)

    grads = (
        grad_means,
        grad_values,
        grad_conics,
        grad_opacities,
        grad_samples,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    return grads

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

class GaussianSampler:
    def __init__(self, debug):
        self.debug = debug

    def preprocess(self, means, values, covariances, conics, opacities, samples):
        debug = self.debug

        num_rendered, binning_buffer, sample_binning_buffer, ranges, sample_ranges = \
            preprocess_gaussians(means, values, covariances, conics, opacities, samples, debug)

        self.means = means
        self.values = values
        self.conics = conics
        self.opacities = opacities
        self.samples = samples
        self.num_rendered = num_rendered
        self.binning_buffer = binning_buffer
        self.sample_binning_buffer = sample_binning_buffer
        self.ranges = ranges
        self.sample_ranges = sample_ranges

    def sample_gaussians(self):
        return sample_gaussians(
            self.means,
            self.values,
            self.conics,
            self.opacities,
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
            self.opacities,
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
            self.opacities,
            self.samples,
            self.num_rendered,
            self.binning_buffer,
            self.sample_binning_buffer,
            self.ranges,
            self.sample_ranges,
            self.debug,
        )
