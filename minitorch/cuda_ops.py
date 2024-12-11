# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions of your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function for CUDA devices"""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """JIT compile a function for CUDA kernels"""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Applies a function element-wise to a tensor."""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Calculate grid and block dimensions
            threads_per_block = THREADS_PER_BLOCK
            blocks_per_grid = (out.size + threads_per_block - 1) // threads_per_block

            # Launch the CUDA kernel
            f[blocks_per_grid, threads_per_block](*out.tuple(), out.size, *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Applies a binary function element-wise to two tensors."""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            # Determine the output shape after broadcasting
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)

            # Calculate grid and block dimensions
            threads_per_block = THREADS_PER_BLOCK
            blocks_per_grid = (out.size + threads_per_block - 1) // threads_per_block

            # Launch the CUDA kernel
            f[blocks_per_grid, threads_per_block](
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduces a tensor along a specified dimension using a binary function."""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            # Calculate the output shape after reduction
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            # Set up grid and block dimensions
            threads_per_block = 1024
            blocks_per_grid = out_a.size

            # Launch the CUDA kernel
            f[blocks_per_grid, threads_per_block](
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )
            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs batched matrix multiplication of two tensors."""
        # Ensure tensors are 3D for batched matrix multiplication
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, *a.shape)
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, *b.shape)
            both_2d += 1
        both_2d = both_2d == 2

        # Determine output shape after broadcasting
        out_shape = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        out_shape.extend([a.shape[-2], b.shape[-1]])
        assert (
            a.shape[-1] == b.shape[-2]
        ), "Inner dimensions must match for matrix multiplication."
        out = a.zeros(tuple(out_shape))

        # Configure grid and block dimensions
        blocks_per_grid = (
            (out.shape[1] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
            (out.shape[2] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threads_per_block = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        # Launch the CUDA kernel
        tensor_matrix_multiply[blocks_per_grid, threads_per_block](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Adjust output shape if original tensors were 2D
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out

    def is_close_zip(self, a: Tensor, b: Tensor) -> Tensor:
        """Performs element-wise comparison to check if tensors are close."""
        from numba import float64

        @cuda.jit()
        def is_close_kernel(
            out_storage: Storage,
            out_shape: Shape,
            out_strides: Strides,
            out_size: int,
            a_storage: Storage,
            a_shape: Shape,
            a_strides: Strides,
            b_storage: Storage,
            b_shape: Shape,
            b_strides: Strides,
        ) -> None:
            out_idx = cuda.local.array(MAX_DIMS, numba.int32)
            a_idx = cuda.local.array(MAX_DIMS, numba.int32)
            b_idx = cuda.local.array(MAX_DIMS, numba.int32)

            idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

            if idx < out_size:
                to_index(idx, out_shape, out_idx)
                broadcast_index(out_idx, out_shape, a_shape, a_idx)
                broadcast_index(out_idx, out_shape, b_shape, b_idx)

                out_pos = index_to_position(out_idx, out_strides)
                a_pos = index_to_position(a_idx, a_strides)
                b_pos = index_to_position(b_idx, b_strides)

                # Explicitly cast to float64
                a_val = float64(a_storage[a_pos])
                b_val = float64(b_storage[b_pos])

                # Compute absolute difference
                diff = a_val - b_val
                if diff < 0.0:
                    diff = -diff

                # Use a float64 tolerance
                tolerance = float64(1e-5)
                if diff <= tolerance:
                    out_storage[out_pos] = 1.0
                else:
                    out_storage[out_pos] = 0.0

        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        threads_per_block = THREADS_PER_BLOCK
        blocks_per_grid = (out.size + threads_per_block - 1) // threads_per_block

        is_close_kernel[blocks_per_grid, threads_per_block](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, int, Storage, Shape, Strides], None]:
    """CUDA-optimized tensor map function that applies a unary function element-wise."""

    def _map(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Allocate local arrays for indices
        out_idx = cuda.local.array(MAX_DIMS, numba.int32)
        in_idx = cuda.local.array(MAX_DIMS, numba.int32)

        # Compute global thread index
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if idx < out_size:
            # Convert linear index to multi-dimensional index
            to_index(idx, out_shape, out_idx)

            # Map output index to input index considering broadcasting
            broadcast_index(out_idx, out_shape, in_shape, in_idx)

            # Calculate positions in storage
            out_pos = index_to_position(out_idx, out_strides)
            in_pos = index_to_position(in_idx, in_strides)

            # Apply the function and store the result
            out_storage[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, Storage, Shape, Strides],
    None,
]:
    """CUDA-optimized tensor zip function that applies a binary function element-wise."""
    # Ensure fn is a device function
    cufn = device_jit(fn)

    def _zip(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Allocate local arrays for indices
        out_idx = cuda.local.array(MAX_DIMS, numba.int32)
        a_idx = cuda.local.array(MAX_DIMS, numba.int32)
        b_idx = cuda.local.array(MAX_DIMS, numba.int32)

        # Compute global thread index
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if idx < out_size:
            # Convert linear index to multi-dimensional index
            to_index(idx, out_shape, out_idx)

            # Map output index to input indices considering broadcasting
            broadcast_index(out_idx, out_shape, a_shape, a_idx)
            broadcast_index(out_idx, out_shape, b_shape, b_idx)

            # Calculate positions in storage
            out_pos = index_to_position(out_idx, out_strides)
            a_pos = index_to_position(a_idx, a_strides)
            b_pos = index_to_position(b_idx, b_strides)

            # Apply the function and store the result
            out_storage[out_pos] = cufn(a_storage[a_pos], b_storage[b_pos])

    # Compile the kernel with the device function known at compile time
    return cuda.jit()(_zip)


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """Practice CUDA kernel for summing elements in blocks."""
    BLOCK_DIM = 32
    shared_cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    thread_idx = cuda.threadIdx.x

    # Load data into shared memory
    if idx < size:
        shared_cache[thread_idx] = a[idx]
    else:
        shared_cache[thread_idx] = 0.0

    # Synchronize threads to ensure all data is loaded
    cuda.syncthreads()

    # Perform parallel reduction within the block
    stride = BLOCK_DIM // 2
    while stride > 0:
        if thread_idx < stride:
            shared_cache[thread_idx] += shared_cache[thread_idx + stride]
        cuda.syncthreads()
        stride //= 2

    # Write the result from each block to the output
    if thread_idx == 0:
        out[cuda.blockIdx.x] = shared_cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Wrapper function to practice summing using CUDA kernel."""
    (size,) = a.shape
    threads_per_block = THREADS_PER_BLOCK
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
    out = TensorData([0.0 for _ in range(blocks_per_grid)], (blocks_per_grid,))
    out.to_cuda_()
    jit_sum_practice[blocks_per_grid, threads_per_block](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, int, float], None
]:
    """CUDA-optimized tensor reduce function that reduces a tensor along a specified dimension."""

    def _reduce(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
        reduce_dim: int,
        start_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        shared_cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        idx = cuda.threadIdx.x
        out_idx = cuda.local.array(MAX_DIMS, numba.int32)
        in_idx = cuda.local.array(MAX_DIMS, numba.int32)

        # Compute the global output index
        out_pos = cuda.blockIdx.x
        if out_pos >= out_size:
            return

        # Convert linear index to multi-dimensional index
        to_index(out_pos, out_shape, out_idx)

        # Initialize reduction result
        shared_cache[idx] = start_value

        # Map output index to input index
        for i in range(len(out_shape)):
            in_idx[i] = out_idx[i]

        # Reduce over the specified dimension
        reduce_size = in_shape[reduce_dim]
        stride = cuda.blockDim.x
        for k in range(idx, reduce_size, stride):
            in_idx[reduce_dim] = k
            in_pos = index_to_position(in_idx, in_strides)
            shared_cache[idx] = fn(shared_cache[idx], in_storage[in_pos])

        # Synchronize threads before reduction
        cuda.syncthreads()

        # Perform parallel reduction in shared memory
        offset = BLOCK_DIM // 2
        while offset > 0:
            if idx < offset:
                shared_cache[idx] = fn(shared_cache[idx], shared_cache[idx + offset])
            cuda.syncthreads()
            offset //= 2

        # Write the reduced result to output
        if idx == 0:
            out_position = index_to_position(out_idx, out_strides)
            out_storage[out_position] = shared_cache[0]

    return cuda.jit()(_reduce)


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice CUDA kernel for matrix multiplication with shared memory."""
    # Thread indices
    row = cuda.threadIdx.x
    col = cuda.threadIdx.y

    # Shared memory for input matrices
    shared_a = cuda.shared.array((THREADS_PER_BLOCK, THREADS_PER_BLOCK), numba.float64)
    shared_b = cuda.shared.array((THREADS_PER_BLOCK, THREADS_PER_BLOCK), numba.float64)

    if row < size and col < size:
        # Load data into shared memory
        shared_a[row, col] = a[row * size + col]
        shared_b[row, col] = b[row * size + col]
    else:
        shared_a[row, col] = 0.0
        shared_b[row, col] = 0.0

    # Synchronize to ensure all data is loaded
    cuda.syncthreads()

    # Compute the dot product for this element
    if row < size and col < size:
        result = 0.0
        for k in range(size):
            result += shared_a[row, k] * shared_b[k, col]

        # Write the result to the output matrix
        out[row * size + col] = result


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Wrapper function for practicing matrix multiplication using CUDA kernel."""
    (size, _) = a.shape
    threads_per_block = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blocks_per_grid = (1, 1)
    out = TensorData([0.0 for _ in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blocks_per_grid, threads_per_block](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out_storage: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA kernel for batched matrix multiplication with shared memory optimization."""
    # Determine batch indices and strides
    batch = cuda.blockIdx.z
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Thread and block indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * cuda.blockDim.y + ty
    col = cuda.blockIdx.x * cuda.blockDim.x + tx

    # Shared memory for tiles
    TILE_SIZE = THREADS_PER_BLOCK
    shared_a = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)
    shared_b = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)

    # Initialize accumulator
    sum_result = 0.0

    # Loop over tiles
    for t in range((a_shape[2] + TILE_SIZE - 1) // TILE_SIZE):
        # Load tiles into shared memory
        if row < a_shape[1] and (t * TILE_SIZE + tx) < a_shape[2]:
            a_pos = (
                batch * a_batch_stride
                + row * a_strides[1]
                + (t * TILE_SIZE + tx) * a_strides[2]
            )
            shared_a[ty, tx] = a_storage[a_pos]
        else:
            shared_a[ty, tx] = 0.0

        if (t * TILE_SIZE + ty) < b_shape[1] and col < b_shape[2]:
            b_pos = (
                batch * b_batch_stride
                + (t * TILE_SIZE + ty) * b_strides[1]
                + col * b_strides[2]
            )
            shared_b[ty, tx] = b_storage[b_pos]
        else:
            shared_b[ty, tx] = 0.0

        # Synchronize to ensure tiles are loaded
        cuda.syncthreads()

        # Perform the multiplication for this tile
        for k in range(TILE_SIZE):
            sum_result += shared_a[ty, k] * shared_b[k, tx]

        # Synchronize before loading next tile
        cuda.syncthreads()

    # Write the result to the output tensor
    if row < out_shape[1] and col < out_shape[2]:
        out_pos = batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]
        out_storage[out_pos] = sum_result


tensor_matrix_multiply = cuda.jit()(_tensor_matrix_multiply)
assert tensor_matrix_multiply is not None
