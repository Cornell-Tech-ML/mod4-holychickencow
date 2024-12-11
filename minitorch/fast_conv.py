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
    """Apply Numba JIT compilation to a function with provided options."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# Pre-JIT key indexing functions for speed.
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
    """Compute a 1D convolution over a batch of input data, using a specified kernel.

    Parameters
    ----------
    output_storage : Storage
        Where the convolution results are stored.
    output_shape : Shape
        The shape of the output tensor: (batch, out_channels, width).
    output_strides : Strides
        Strides for indexing into the output.
    total_elements : int
        Total number of elements in the output tensor.
    input_storage : Storage
        The underlying data for the input tensor.
    input_shape : Shape
        The shape of the input: (batch, in_channels, width).
    input_strides : Strides
        Strides for indexing into the input.
    kernel_storage : Storage
        The underlying data for the kernel tensor.
    kernel_shape : Shape
        The shape of the kernel: (out_channels, in_channels, kernel_width).
    kernel_strides : Strides
        Strides for indexing into the kernel.
    reverse_order : bool
        Whether to reverse the direction of convolution indexing.

    Notes
    -----
    This performs a per-element convolution operation. For each position,
    it accumulates the product of corresponding input and kernel values.

    """
    batch_count, out_ch, out_w = output_shape
    in_b, in_ch, in_w = input_shape
    ker_out_ch, ker_in_ch, ker_w = kernel_shape

    # Validate dimension consistency
    assert in_b == batch_count and in_ch == ker_in_ch and out_ch == ker_out_ch

    in_s = input_strides
    ker_s = kernel_strides
    out_s = output_strides

    for b in prange(batch_count):
        for oc in prange(out_ch):
            for ow in prange(out_w):
                accum = 0.0
                for ic in prange(in_ch):
                    for kw in prange(ker_w):
                        if reverse_order:
                            inp_w_pos = ow - kw
                        else:
                            inp_w_pos = ow + kw
                        if 0 <= inp_w_pos < in_w:
                            val_inp = input_storage[b * in_s[0] + ic * in_s[1] + inp_w_pos * in_s[2]]
                            val_ker = kernel_storage[oc * ker_s[0] + ic * ker_s[1] + kw * ker_s[2]]
                            accum += val_inp * val_ker
                output_storage[b * out_s[0] + oc * out_s[1] + ow * out_s[2]] = accum


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, kernel_tensor: Tensor) -> Tensor:
        """Forward pass for 1D convolution.

        Parameters
        ----------
        ctx : Context
            Context to store values for backward computation.
        input_tensor : Tensor
            Input data of shape (batch, in_channels, width).
        kernel_tensor : Tensor
            Kernel filter of shape (out_channels, in_channels, kernel_width).

        Returns
        -------
        Tensor
            Convolved output of shape (batch, out_channels, width).

        """
        ctx.save_for_backward(input_tensor, kernel_tensor)
        b, in_c, w = input_tensor.shape
        out_c, ker_in_c, ker_w = kernel_tensor.shape
        assert in_c == ker_in_c

        result = input_tensor.zeros((b, out_c, w))
        tensor_conv1d(
            *result.tuple(), result.size,
            *input_tensor.tuple(),
            *kernel_tensor.tuple(),
            False
        )
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for 1D convolution.

        Parameters
        ----------
        ctx : Context
            Context with saved tensors from forward pass.
        grad_output : Tensor
            Gradient of the loss with respect to the convolution output.

        Returns
        -------
        (Tensor, Tensor)
            Gradients with respect to input and kernel.

        """
        input_tensor, kernel_tensor = ctx.saved_values
        b, in_c, w = input_tensor.shape
        out_c, ker_in_c, ker_w = kernel_tensor.shape

        # Compute grad_kernel
        grad_kernel = grad_output.zeros((in_c, out_c, ker_w))
        # For grad_kernel, we convolve input^T with grad_output^T
        inp_trans = input_tensor.permute(1, 0, 2)
        grad_out_trans = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_kernel.tuple(), grad_kernel.size,
            *inp_trans.tuple(),
            *grad_out_trans.tuple(),
            False
        )
        grad_kernel = grad_kernel.permute(1, 0, 2)

        # Compute grad_input
        grad_input = input_tensor.zeros((b, in_c, w))
        ker_trans = kernel_tensor.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(), grad_input.size,
            *grad_output.tuple(),
            *ker_trans.tuple(),
            True
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
    """Compute a 2D convolution over a batch of inputs using a given kernel.

    Parameters
    ----------
    output_storage : Storage
        Storage for output tensor.
    output_shape : Shape
        Shape of output (batch, out_channels, height, width).
    output_strides : Strides
        Strides for output indexing.
    total_elements : int
        Number of output elements.
    input_storage : Storage
        Input data storage.
    input_shape : Shape
        Input shape (batch, in_channels, height, width).
    input_strides : Strides
        Strides for input indexing.
    kernel_storage : Storage
        Kernel data storage.
    kernel_shape : Shape
        Kernel shape (out_channels, in_channels, kernel_h, kernel_w).
    kernel_strides : Strides
        Strides for kernel indexing.
    reverse_order : bool
        Whether to index the kernel in reverse order.

    Notes
    -----
    This loops over every element of the output and computes the
    sum of element-wise multiplications between a patch of the input
    and the kernel.

    """
    b_out, ch_out, h_out, w_out = output_shape
    b_in, ch_in, h_in, w_in = input_shape
    ker_ch, ker_in_ch, ker_h, ker_w = kernel_shape

    # Check shape consistency
    assert b_in == b_out and ch_in == ker_in_ch and ch_out == ker_ch

    in_s = input_strides
    ker_s = kernel_strides
    out_s = output_strides

    for b in prange(b_out):
        for oc in prange(ch_out):
            for oh in prange(h_out):
                for ow in prange(w_out):
                    total = 0.0
                    for ic in prange(ch_in):
                        for kh in prange(ker_h):
                            for kw in prange(ker_w):
                                if reverse_order:
                                    ih = oh + kh - ker_h + 1
                                    iw = ow + kw - ker_w + 1
                                else:
                                    ih = oh + kh
                                    iw = ow + kw
                                if 0 <= ih < h_in and 0 <= iw < w_in:
                                    val_in = input_storage[
                                        b * in_s[0]
                                        + ic * in_s[1]
                                        + ih * in_s[2]
                                        + iw * in_s[3]
                                    ]
                                    val_k = kernel_storage[
                                        oc * ker_s[0]
                                        + ic * ker_s[1]
                                        + kh * ker_s[2]
                                        + kw * ker_s[3]
                                    ]
                                    total += val_in * val_k
                    output_storage[
                        b * out_s[0] + oc * out_s[1] + oh * out_s[2] + ow * out_s[3]
                    ] = total


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, kernel_tensor: Tensor) -> Tensor:
        """Forward pass for 2D convolution.

        Parameters
        ----------
        ctx : Context
            Context for saving values for backward.
        input_tensor : Tensor
            Input of shape (batch, in_channels, height, width).
        kernel_tensor : Tensor
            Kernel of shape (out_channels, in_channels, kernel_h, kernel_w).

        Returns
        -------
        Tensor
            Output of shape (batch, out_channels, height, width).

        """
        ctx.save_for_backward(input_tensor, kernel_tensor)
        b, in_c, h, w = input_tensor.shape
        out_c, ker_in_c, ker_h, ker_w = kernel_tensor.shape
        assert in_c == ker_in_c

        out = input_tensor.zeros((b, out_c, h, w))
        tensor_conv2d(
            *out.tuple(), out.size,
            *input_tensor.tuple(),
            *kernel_tensor.tuple(),
            False
        )
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for 2D convolution.

        Parameters
        ----------
        ctx : Context
            Context with saved tensors from forward.
        grad_output : Tensor
            Gradient wrt the convolution output.

        Returns
        -------
        (Tensor, Tensor)
            Gradients wrt the input and the kernel.
            
        """
        input_tensor, kernel_tensor = ctx.saved_values
        b, in_c, h, w = input_tensor.shape
        out_c, ker_in_c, ker_h, ker_w = kernel_tensor.shape

        # Compute grad wrt kernel
        grad_kernel = grad_output.zeros((in_c, out_c, ker_h, ker_w))
        inp_t = input_tensor.permute(1, 0, 2, 3)
        grad_out_t = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_kernel.tuple(), grad_kernel.size,
            *inp_t.tuple(),
            *grad_out_t.tuple(),
            False
        )
        grad_kernel = grad_kernel.permute(1, 0, 2, 3)

        # Compute grad wrt input
        grad_input = input_tensor.zeros((b, in_c, h, w))
        ker_t = kernel_tensor.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(), grad_input.size,
            *grad_output.tuple(),
            *ker_t.tuple(),
            True
        )

        return grad_input, grad_kernel


conv2d = Conv2dFun.apply
