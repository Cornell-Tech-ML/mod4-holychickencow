from typing import Tuple, TypeVar, Any

from numba import njit as _njit
from numba import prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to apply Numba's JIT compilation with specified options."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# JIT compile essential tensor_data functions for performance.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of shape (batch, in_channels, width) and weight tensor
    of shape (out_channels, in_channels, k_width), computes an output of
    shape (batch, out_channels, width).

    Args:
        out (Storage): storage for output tensor.
        out_shape (Shape): shape of the output tensor.
        out_strides (Strides): strides of the output tensor.
        out_size (int): number of elements in the output tensor.
        input (Storage): storage for the input tensor.
        input_shape (Shape): shape of the input tensor.
        input_strides (Strides): strides of the input tensor.
        weight (Storage): storage for the weight tensor.
        weight_shape (Shape): shape of the weight tensor.
        weight_strides (Strides): strides of the weight tensor.
        reverse (bool): Whether to anchor the kernel at the left (False) or right (True).

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape
    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    s3 = out_strides

    for b in prange(batch):
        for oc in prange(out_channels):
            for ow in prange(out_width):
                acc = 0.0
                for ic in prange(in_channels):
                    for kw_ in prange(kw):
                        iw = ow - kw_ if reverse else ow + kw_
                        if 0 <= iw < width:
                            acc += (
                                input[b * s1[0] + ic * s1[1] + iw * s1[2]]
                                * weight[oc * s2[0] + ic * s2[1] + kw_ * s2[2]]
                            )
                out[b * s3[0] + oc * s3[1] + ow * s3[2]] = acc


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Perform a 1D convolution forward pass.

        Args:
            ctx (Context): Context for saving data for backward pass.
            input (Tensor): Input tensor (batch, in_channels, width).
            weight (Tensor): Weight tensor (out_channels, in_channels, k_width).

        Returns:
            Tensor: The result of the convolution (batch, out_channels, width).

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels_w, kw = weight.shape
        assert in_channels == in_channels_w

        # Allocate output
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(# type: ignore
            *output.tuple(),
            output.size,
            *input.tuple(),
            *weight.tuple(),
            False,# type: ignore
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients for 1D convolution."""
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels_w, kw = weight.shape
        assert in_channels == in_channels_w

        # Gradient wrt weight
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(# type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,# type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        # Gradient wrt input
        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(# type: ignore
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,# type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input (batch, in_channels, height, width) and weight
    (out_channels, in_channels, k_height, k_width), produces output
    (batch, out_channels, height, width).

    Args:
        out (Storage): Output storage.
        out_shape (Shape): Output shape.
        out_strides (Strides): Output strides.
        out_size (int): Number of elements in the output tensor.
        input (Storage): Input storage.
        input_shape (Shape): Input shape.
        input_strides (Strides): Input strides.
        weight (Storage): Weight storage.
        weight_shape (Shape): Weight shape.
        weight_strides (Strides): Weight strides.
        reverse (bool): Whether to anchor kernel top-left (False) or bottom-right (True).

    """
    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape
    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    s3 = out_strides
    s10, s11, s12, s13 = s1
    s20, s21, s22, s23 = s2
    s30, s31, s32, s33 = s3

    for b in prange(batch):
        for oc in prange(out_channels):
            for oh in prange(out_height):
                for ow in prange(out_width):
                    acc = 0.0
                    for ic in prange(in_channels):
                        for kh_ in prange(kh):
                            for kw_ in prange(kw):
                                if reverse:
                                    ih = oh + kh_ - kh + 1
                                    iw = ow + kw_ - kw + 1
                                else:
                                    ih = oh + kh_
                                    iw = ow + kw_
                                if 0 <= ih < height and 0 <= iw < width:
                                    acc += (
                                        input[b * s10 + ic * s11 + ih * s12 + iw * s13]
                                        * weight[oc * s20 + ic * s21 + kh_ * s22 + kw_ * s23]
                                    )
                    out[b * s30 + oc * s31 + oh * s32 + ow * s33] = acc


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, kernel_tensor: Tensor) -> Tensor:  # noqa: D102
        ctx.save_for_backward(input_tensor, kernel_tensor)
        batch, in_channels, height, width = input_tensor.shape
        out_channels, in_channels_k, kernel_h, kernel_w = kernel_tensor.shape
        assert in_channels == in_channels_k

        output = input_tensor.zeros((batch, out_channels, height, width))
        tensor_conv2d(
            output.storage,
            output.shape,
            output.strides,
            output.size,
            input_tensor.storage,
            input_tensor.shape,
            input_tensor.strides,
            kernel_tensor.storage,
            kernel_tensor.shape,
            kernel_tensor.strides,
            False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:  # noqa: D102
        input_tensor, kernel_tensor = ctx.saved_values
        batch, in_channels, height, width = input_tensor.shape
        out_channels, in_channels_k, kernel_h, kernel_w = kernel_tensor.shape

        grad_kernel = grad_output.zeros((in_channels, out_channels, kernel_h, kernel_w))
        transposed_input = input_tensor.permute(1, 0, 2, 3)
        transposed_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            grad_kernel.storage,
            grad_kernel.shape,
            grad_kernel.strides,
            grad_kernel.size,
            transposed_input.storage,
            transposed_input.shape,
            transposed_input.strides,
            transposed_grad_output.storage,
            transposed_grad_output.shape,
            transposed_grad_output.strides,
            False
        )
        grad_kernel = grad_kernel.permute(1, 0, 2, 3)

        grad_input = input_tensor.zeros((batch, in_channels, height, width))
        transposed_kernel = kernel_tensor.permute(1, 0, 2, 3)
        tensor_conv2d(
            grad_input.storage,
            grad_input.shape,
            grad_input.strides,
            grad_input.size,
            grad_output.storage,
            grad_output.shape,
            grad_output.strides,
            transposed_kernel.storage,
            transposed_kernel.shape,
            transposed_kernel.strides,
            True
        )
        return grad_input, grad_kernel


conv2d = Conv2dFun.apply
