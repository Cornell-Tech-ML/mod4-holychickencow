from typing import Tuple

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling.

    This breaks the input (batch x channel x height x width) into a tiled form:
    (batch x channel x new_height x new_width x (kernel_height * kernel_width)).

    Args:
    ----
        input: Tensor of shape (B, C, H, W)
        kernel: Tuple (kh, kw) representing pooling kernel size

    Returns:
    -------
        A tuple (tiled_tensor, new_h, new_w) where:
        - tiled_tensor: shape (B, C, new_h, new_w, kh*kw)
        - new_h: integer, height after pooling
        - new_w: integer, width after pooling

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel

    # Validate that height and width are divisible by the kernel dimensions
    assert height % kh == 0, "Height not divisible by kernel height."
    assert width % kw == 0, "Width not divisible by kernel width."

    # Compute new dimensions after applying the kernel
    new_h = height // kh
    new_w = width // kw
    tile_count = kh * kw

    # First, create a contiguous view that separates out the kernel along height and width
    # View as (batch, channel, new_h, kh, new_w, kw)
    viewed = input.contiguous().view(batch, channel, new_h, kh, new_w, kw)

    # Rearrange (permute) dimensions to group kernel elements together at the end
    # Permute to (batch, channel, new_h, new_w, kh, kw)
    reordered = viewed.permute(0, 1, 2, 4, 3, 5)

    # Make contiguous again before the final view
    final = reordered.contiguous().view(batch, channel, new_h, new_w, tile_count)

    return final, new_h, new_w


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform average pooling by tiling and then taking the mean over the tile dimension.

    Args:
    ----
        input: Tensor of shape (B, C, H, W)
        kernel: (kh, kw)

    Returns:
    -------
        Tensor of shape (B, C, new_h, new_w) after average pooling.

    """
    tiled, nh, nw = tile(input, kernel)
    return tiled.mean(-1).contiguous().view(tiled.shape[0], tiled.shape[1], nh, nw)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Max forward

        Performs a max-reduction along a given dimension.
        """
        dimension = int(dim.item())
        maximum = a.f.max_reduce(a, dimension)
        mask = a.f.eq_zip(a, maximum)
        ctx.save_for_backward(mask)
        return maximum

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Max backward

        Backpropagates the gradient only where the maximum values were selected.
        """
        (mask,) = ctx.saved_values
        return mask * grad_output, 0.0


def max(input: Tensor, dim: int | None = None) -> Tensor:
    """Compute the max along a specified dimension.

    If dim is None, flattens the input and takes the max over all elements.

    Args:
    ----
        input: Tensor
        dim: int or None

    Returns:
    -------
        Tensor with the dimension reduced by taking the maximum.

    """
    if dim is None:
        # Flatten input and take max over all elements
        flat = input.contiguous().view(input.size)
        return Max.apply(flat, input._ensure_tensor(0))
    return Max.apply(input, input._ensure_tensor(dim))


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform max pooling by splitting into tiles and taking the max over the tile dimension.

    Args:
    ----
        input: Tensor of shape (B, C, H, W)
        kernel: (kh, kw)

    Returns:
    -------
        Tensor of shape (B, C, new_h, new_w) after max pooling.

    """
    tiled, nh, nw = tile(input, kernel)
    return max(tiled, 4).contiguous().view(tiled.shape[0], tiled.shape[1], nh, nw)


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax along a specified dimension.

    Args:
    ----
        input: Tensor
        dim: dimension to apply softmax along

    Returns:
    -------
        Tensor with softmax probabilities along that dimension.

    """
    exp_vals = input.exp()
    return exp_vals / exp_vals.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute log-softmax along a specified dimension using the log-sum-exp trick.

    Args:
    ----
        input: Tensor
        dim: dimension to apply log-softmax along

    Returns:
    -------
        Tensor with log-softmax applied along that dimension.

    """
    max_vals = max(input, dim)
    shifted = input - max_vals
    sum_exps = shifted.exp().sum(dim)
    log_sum_exps = sum_exps.log() + max_vals
    return input - log_sum_exps


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor.

    Args:
    ----
        input: Tensor
        p: dropout probability
        ignore: if True, do nothing

    Returns:
    -------
        Tensor with elements dropped out according to probability p.

    """
    if ignore:
        return input
    if p == 0.0:
        return input
    if p == 1.0:
        return input.zeros()

    mask = rand(input.shape, backend=input.backend, requires_grad=False) > p
    return input * mask
