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

def sample_gaussians(
    means,
    values,
    covariances,
    conics,
    opacities,
    samples,
    debug,
):
    return _SampleGaussians.apply(
        means,
        values,
        covariances,
        conics,
        opacities,
        samples,
        debug,
    )

class _SampleGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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

        # Invoke C++/CUDA sampler
        if debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, out_values, binning_buffer, sample_binning_buffer, ranges, sample_ranges = _C.sample_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, out_values, binning_buffer, sample_binning_buffer, ranges, sample_ranges = _C.sample_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.debug = debug
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means, values, conics, opacities, samples, binning_buffer, sample_binning_buffer, ranges, sample_ranges)
        return out_values

    @staticmethod
    def backward(ctx, grad_out):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        debug = ctx.debug
        means, values, conics, opacities, samples, binning_buffer, sample_binning_buffer, ranges, sample_ranges = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (means,
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
                debug)

        # Compute gradients for relevant tensors by invoking backward method
        if debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means, grad_values, grad_conics, grad_opacities, grad_samples = _C.sample_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means, grad_values, grad_conics, grad_opacities, grad_samples = _C.sample_gaussians_backward(*args)

        grads = (
            grad_means,
            grad_values,
            None,
            grad_conics,
            grad_opacities,
            grad_samples,
            None,
        )

        return grads

class GaussianSampler(nn.Module):
    def __init__(self, debug):
        super().__init__()
        self.debug = debug

    def forward(self, means, values, covariances, conics, opacities, samples):
        debug = self.debug

        # Invoke C++/CUDA rasterization routine
        return sample_gaussians(
            means,
            values,
            covariances,
            conics,
            opacities,
            samples,
            debug,
        )

