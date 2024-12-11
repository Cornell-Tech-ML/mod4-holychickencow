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
    output_storage: Storage,
    output_shape: Shape,
    output_strides: Strides,
    total_elements: int,
    input_storage: Storage,
    input_shape: Shape,
    input_strides: Strides,
    kernel_storage: Storage,
    kernel_shape: Shape,
    kernel_strides: Strides,
    reverse_order: bool,
) -> None:
    """Implementation of 1D Convolution using nested parallel loops.

    Args:
    ----
        output_storage (Storage): Storage for the output tensor.
        output_shape (Shape): Shape of the output tensor.
        output_strides (Strides): Strides of the output tensor.
        total_elements (int): Total number of elements in the output tensor.
        input_storage (Storage): Storage for the input tensor.
        input_shape (Shape): Shape of the input tensor.
        input_strides (Strides): Strides of the input tensor.
        kernel_storage (Storage): Storage for the kernel tensor.
        kernel_shape (Shape): Shape of the kernel tensor.
        kernel_strides (Strides): Strides of the kernel tensor.
        reverse_order (bool): Flag to determine kernel alignment.

    """
    out_batch, out_ch, out_width = output_shape
    in_batch, in_ch, in_width = input_shape
    kernel_ch, kernel_in_ch, kernel_width = kernel_shape

    # Ensure tensor dimensions match for convolution
    assert in_batch == out_batch and in_ch == kernel_in_ch and out_ch == kernel_ch

    in_s = input_strides
    ker_s = kernel_strides
    out_s = output_strides

    for b in prange(out_batch):
        for oc in prange(out_ch):
            for ow in prange(out_width):
                accumulator = 0.0
                for ic in prange(in_ch):
                    for kw in prange(kernel_width):
                        if reverse_order:
                            input_idx = ow - kw
                        else:
                            input_idx = ow + kw
                        if 0 <= input_idx < in_width:
                            input_val = input_storage[
                                b * in_s[0] + ic * in_s[1] + input_idx * in_s[2]
                            ]
                            kernel_val = kernel_storage[
                                oc * ker_s[0] + ic * ker_s[1] + kw * ker_s[2]
                            ]
                            accumulator += input_val * kernel_val
                output_storage[b * out_s[0] + oc * out_s[1] + ow * out_s[2]] = (
                    accumulator
                )


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, kernel_tensor: Tensor) -> Tensor:
        """Performs the forward pass of a 1D Convolution.

        Args:
        ----
            ctx (Context): Context to save tensors for backward pass.
            input_tensor (Tensor): Input tensor with shape (batch, in_channels, width).
            kernel_tensor (Tensor): Kernel tensor with shape (out_channels, in_channels, kernel_width).

        Returns:
        -------
            Tensor: Output tensor after convolution with shape (batch, out_channels, width).

        """
        ctx.save_for_backward(input_tensor, kernel_tensor)
        batch, in_channels, width = input_tensor.shape
        out_channels, in_channels_k, kernel_width = kernel_tensor.shape
        assert in_channels == in_channels_k

        # Initialize the output tensor with zeros
        output = input_tensor.zeros((batch, out_channels, width))

        # Execute the convolution
        tensor_conv1d(
            *output.tuple(),
            output.size,
            *input_tensor.tuple(),
            *kernel_tensor.tuple(),
            False,
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes gradients for the 1D Convolution.

        Args:
        ----
            ctx (Context): Context containing saved tensors.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to input and kernel tensors.

        """
        input_tensor, kernel_tensor = ctx.saved_values
        batch, in_channels, width = input_tensor.shape
        out_channels, in_channels_k, kernel_width = kernel_tensor.shape

        grad_kernel = grad_output.zeros((in_channels, out_channels, kernel_width))
        transposed_input = input_tensor.permute(1, 0, 2)
        transposed_grad_output = grad_output.permute(1, 0, 2)

        tensor_conv1d(
            *grad_kernel.tuple(),
            grad_kernel.size,
            *transposed_input.tuple(),
            *transposed_grad_output.tuple(),
            False,
        )
        grad_kernel = grad_kernel.permute(1, 0, 2)

        grad_input = input_tensor.zeros((batch, in_channels, width))
        transposed_kernel = kernel_tensor.permute(1, 0, 2)

        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *transposed_kernel.tuple(),
            True,
        )
        return grad_input, grad_kernel


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    output_storage: Storage,
    output_shape: Shape,
    output_strides: Strides,
    total_elements: int,
    input_storage: Storage,
    input_shape: Shape,
    input_strides: Strides,
    kernel_storage: Storage,
    kernel_shape: Shape,
    kernel_strides: Strides,
    reverse_order: bool,
) -> None:
    """Implementation of 2D Convolution using nested parallel loops.

    Args:
    ----
        output_storage (Storage): Storage for the output tensor.
        output_shape (Shape): Shape of the output tensor.
        output_strides (Strides): Strides of the output tensor.
        total_elements (int): Total number of elements in the output tensor.
        input_storage (Storage): Storage for the input tensor.
        input_shape (Shape): Shape of the input tensor.
        input_strides (Strides): Strides of the input tensor.
        kernel_storage (Storage): Storage for the kernel tensor.
        kernel_shape (Shape): Shape of the kernel tensor.
        kernel_strides (Strides): Strides of the kernel tensor.
        reverse_order (bool): Flag to determine kernel alignment.

    """
    out_batch, out_ch, out_h, out_w = output_shape
    in_batch, in_ch, in_h, in_w = input_shape
    kernel_ch, kernel_in_ch, kernel_h, kernel_w = kernel_shape

    # Validate tensor dimensions for convolution
    assert in_batch == out_batch and in_ch == kernel_in_ch and out_ch == kernel_ch

    in_s = input_strides
    ker_s = kernel_strides
    out_s = output_strides

    for b in prange(out_batch):
        for oc in prange(out_ch):
            for oh in prange(out_h):
                for ow in prange(out_w):
                    accumulator = 0.0
                    for ic in prange(in_ch):
                        for kh in prange(kernel_h):
                            for kw in prange(kernel_w):
                                if reverse_order:
                                    ih = oh + kh - kernel_h + 1
                                    iw = ow + kw - kernel_w + 1
                                else:
                                    ih = oh + kh
                                    iw = ow + kw
                                if 0 <= ih < in_h and 0 <= iw < in_w:
                                    input_idx = (
                                        b * in_s[0]
                                        + ic * in_s[1]
                                        + ih * in_s[2]
                                        + iw * in_s[3]
                                    )
                                    kernel_idx = (
                                        oc * ker_s[0]
                                        + ic * ker_s[1]
                                        + kh * ker_s[2]
                                        + kw * ker_s[3]
                                    )
                                    accumulator += (
                                        input_storage[input_idx]
                                        * kernel_storage[kernel_idx]
                                    )
                    output_idx = (
                        b * out_s[0] + oc * out_s[1] + oh * out_s[2] + ow * out_s[3]
                    )
                    output_storage[output_idx] = accumulator


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, kernel_tensor: Tensor) -> Tensor:
        """Performs the forward pass of a 2D Convolution.

        Args:
        ----
            ctx (Context): Context to save tensors for backward pass.
            input_tensor (Tensor): Input tensor with shape (batch, in_channels, height, width).
            kernel_tensor (Tensor): Kernel tensor with shape (out_channels, in_channels, k_height, k_width).

        Returns:
        -------
            Tensor: Output tensor after convolution with shape (batch, out_channels, height, width).

        """
        ctx.save_for_backward(input_tensor, kernel_tensor)
        batch, in_channels, height, width = input_tensor.shape
        out_channels, in_channels_k, kernel_h, kernel_w = kernel_tensor.shape
        assert in_channels == in_channels_k

        # Initialize the output tensor with zeros
        output = input_tensor.zeros((batch, out_channels, height, width))

        # Execute the convolution
        tensor_conv2d(
            *output.tuple(),
            output.size,
            *input_tensor.tuple(),
            *kernel_tensor.tuple(),
            False,
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes gradients for the 2D Convolution.

        Args:
        ----
            ctx (Context): Context containing saved tensors.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to input and kernel tensors.

        """
        input_tensor, kernel_tensor = ctx.saved_values
        batch, in_channels, height, width = input_tensor.shape
        out_channels, in_channels_k, kernel_h, kernel_w = kernel_tensor.shape

        grad_kernel = grad_output.zeros((in_channels, out_channels, kernel_h, kernel_w))
        transposed_input = input_tensor.permute(1, 0, 2, 3)
        transposed_grad_output = grad_output.permute(1, 0, 2, 3)

        tensor_conv2d(
            *grad_kernel.tuple(),
            grad_kernel.size,
            *transposed_input.tuple(),
            *transposed_grad_output.tuple(),
            False,
        )
        grad_kernel = grad_kernel.permute(1, 0, 2, 3)

        grad_input = input_tensor.zeros((batch, in_channels, height, width))
        transposed_kernel = kernel_tensor.permute(1, 0, 2, 3)

        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *transposed_kernel.tuple(),
            True,
        )
        return grad_input, grad_kernel


conv2d = Conv2dFun.apply
