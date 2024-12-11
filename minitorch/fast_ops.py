from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

from numba import njit as _njit, prange
import numpy as np

from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1 to run these tests without JIT.

# This code will JIT compile fast versions of your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Just-in-time compile a function using Numba."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See tensor_ops.py"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See tensor_ops.py"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See tensor_ops.py"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Initialize output tensor with the starting value.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a: The first input tensor.
            b: The second input tensor.

        Returns:
        -------
            A new tensor containing the result of the matrix multiplication.

        """
        # Ensure both tensors are 3-dimensional
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        # Determine the output shape after broadcasting
        output_shape = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        output_shape.append(a.shape[-2])
        output_shape.append(b.shape[-1])
        assert (
            a.shape[-1] == b.shape[-2]
        ), "Matrix dimensions do not match for multiplication."
        out = a.zeros(tuple(output_shape))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # If both original tensors were 2D, adjust the output shape accordingly
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA optimized tensor map function.

    Applies a unary function element-wise to the input tensor, supporting broadcasting.

    Optimizations:

    * Main loop is parallelized.
    * Utilizes numpy arrays for indices to enhance performance.
    * Skips indexing when output and input tensors are stride-aligned.

    Args:
    ----
        fn: A unary function to apply to each element.

    Returns:
    -------
        A function that performs the element-wise operation.

    """

    def _map(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Check for stride alignment and shape equality
        if (
            len(out_strides) == len(in_strides)
            and np.array_equal(out_strides, in_strides)
            and np.array_equal(out_shape, in_shape)
        ):
            # Directly apply the function without indexing
            for idx in prange(len(out_storage)):
                out_storage[idx] = fn(in_storage[idx])
            return

        # General case with indexing
        total_elements = np.prod(out_shape)

        for idx in prange(total_elements):
            # Initialize index arrays
            out_idx = np.empty(len(out_shape), dtype=np.int32)
            in_idx = np.empty(len(in_shape), dtype=np.int32)

            # Convert linear index to multi-dimensional index
            to_index(idx, out_shape, out_idx)

            # Compute positions in storage
            out_pos = index_to_position(out_idx, out_strides)
            broadcast_index(out_idx, out_shape, in_shape, in_idx)
            in_pos = index_to_position(in_idx, in_strides)

            # Apply the function
            out_storage[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA optimized tensor zip function.

    Applies a binary function element-wise to two input tensors, supporting broadcasting.

    Optimizations:

    * Main loop is parallelized.
    * Utilizes numpy arrays for indices to enhance performance.
    * Skips indexing when all tensors are stride-aligned.

    Args:
    ----
        fn: A binary function to apply to each pair of elements.

    Returns:
    -------
        A function that performs the element-wise binary operation.

    """

    def _zip(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Check for stride alignment and shape equality
        if (
            len(out_strides) == len(a_strides) == len(b_strides)
            and np.array_equal(out_shape, a_shape)
            and np.array_equal(out_shape, b_shape)
            and np.array_equal(out_strides, a_strides)
            and np.array_equal(out_strides, b_strides)
        ):
            # Directly apply the function without indexing
            for idx in prange(len(out_storage)):
                out_storage[idx] = fn(a_storage[idx], b_storage[idx])
            return

        # General case with indexing
        total_elements = np.prod(out_shape)

        for idx in prange(total_elements):
            # Initialize index arrays
            out_idx = np.empty(len(out_shape), dtype=np.int32)
            a_idx = np.empty(len(a_shape), dtype=np.int32)
            b_idx = np.empty(len(b_shape), dtype=np.int32)

            # Convert linear index to multi-dimensional index
            to_index(idx, out_shape, out_idx)

            # Compute positions in storage
            out_pos = index_to_position(out_idx, out_strides)
            broadcast_index(out_idx, out_shape, a_shape, a_idx)
            a_pos = index_to_position(a_idx, a_strides)
            broadcast_index(out_idx, out_shape, b_shape, b_idx)
            b_pos = index_to_position(b_idx, b_strides)

            # Apply the function
            out_storage[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA optimized tensor reduce function.

    Reduces a tensor along a specified dimension using a binary function.

    Optimizations:

    * Main loop is parallelized.
    * Utilizes numpy arrays for indices to enhance performance.
    * Inner loop avoids global writes and function calls.

    Args:
    ----
        fn: A binary function to reduce elements.
        reduce_dim: The dimension along which to reduce.

    Returns:
    -------
        A function that performs the reduction.

    """

    def _reduce(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # Compute total number of output elements
        total_output_elements = np.prod(out_shape)

        # Parallel loop over output elements
        for idx in prange(total_output_elements):
            # Initialize index arrays
            out_idx = np.empty(len(out_shape), dtype=np.int32)
            in_idx = np.empty(len(in_shape), dtype=np.int32)

            # Convert linear index to multi-dimensional index
            to_index(idx, out_shape, out_idx)

            # Compute position in output storage
            out_pos = index_to_position(out_idx, out_strides)

            # Set up initial index for input tensor
            for dim in range(len(out_shape)):
                in_idx[dim] = out_idx[dim]

            # Initialize reduction result with the first element along the reduced dimension
            in_idx[reduce_dim] = 0
            in_pos = index_to_position(in_idx, in_strides)
            result = in_storage[in_pos]

            # Iterate over the reduction dimension
            for i in range(1, in_shape[reduce_dim]):
                in_idx[reduce_dim] = i
                in_pos = index_to_position(in_idx, in_strides)
                result = fn(result, in_storage[in_pos])

            # Store the reduced value
            out_storage[out_pos] = result

    return njit(_reduce, parallel=True)


def _tensor_matrix_multiply(
    out_storage: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA optimized tensor matrix multiplication function.

    Performs batched matrix multiplication of two tensors, supporting broadcasting.

    Should work for any tensor shapes that broadcast as long as:

        assert a_shape[-1] == b_shape[-2]

    Optimizations:

    * Outer loops are parallelized.
    * Avoids index buffers and function calls within inner loops.
    * Inner loop performs a single multiply per iteration.

    Args:
    ----
        out_storage: Storage for the output tensor.
        out_shape: Shape of the output tensor.
        out_strides: Strides of the output tensor.
        a_storage: Storage for the first input tensor.
        a_shape: Shape of the first input tensor.
        a_strides: Strides of the first input tensor.
        b_storage: Storage for the second input tensor.
        b_shape: Shape of the second input tensor.
        b_strides: Strides of the second input tensor.

    Returns:
    -------
        None. The result is stored in `out_storage`.

    """
    # Determine batch strides, considering broadcasting
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Parallelize over batches and rows
    for batch_idx in prange(out_shape[0]):
        for i in range(out_shape[1]):
            for j in range(out_shape[2]):
                # Compute position in output storage
                out_pos = (
                    batch_idx * out_strides[0] + i * out_strides[1] + j * out_strides[2]
                )

                # Initialize accumulator
                sum_result = 0.0

                # Perform the dot product
                for k in range(a_shape[2]):
                    a_pos = (
                        batch_idx * a_batch_stride + i * a_strides[1] + k * a_strides[2]
                    )
                    b_pos = (
                        batch_idx * b_batch_stride + k * b_strides[1] + j * b_strides[2]
                    )
                    sum_result += a_storage[a_pos] * b_storage[b_pos]

                # Store the computed value
                out_storage[out_pos] = sum_result


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
