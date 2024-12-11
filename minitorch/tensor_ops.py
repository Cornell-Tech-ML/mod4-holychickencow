from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce placeholder"""
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns:
        -------
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.max_reduce = ops.reduce(operators.max, float("-inf"))
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
            :class:`TensorData` : new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            start: starting value for reduction
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
        -------
            :class:`TensorData` : new tensor

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Applies a function element-wise to a tensor.

    Args:
    ----
        fn (Callable[[float], float]): The function to apply to each element.

    Returns:
    -------
        Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]: The mapped tensor.

    """

    def _map(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        total_elements = 1
        for dim_size in out_shape:
            total_elements *= dim_size

        for pos in range(total_elements):
            current_out_idx: Index = np.zeros(len(out_shape), dtype=np.int32)
            to_index(pos, out_shape, current_out_idx)
            current_in_idx: Index = np.zeros(len(in_shape), dtype=np.int32)
            broadcast_index(current_out_idx, out_shape, in_shape, current_in_idx)
            out_pos = index_to_position(current_out_idx, out_strides)
            in_pos = index_to_position(current_in_idx, in_strides)
            out_storage[out_pos] = fn(in_storage[in_pos])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [
        Storage,
        Shape,
        Strides,
        Storage,
        Shape,
        Strides,
        Storage,
        Shape,
        Strides,
    ],
    None,
]:
    """Combines two tensors element-wise using a binary function.

    Args:
    ----
        fn (Callable[[float, float], float]): The function to apply to each pair of elements.

    Returns:
    -------
        Callable[[Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None]: The zipped tensor.

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
        total_elements = 1
        for dim_size in out_shape:
            total_elements *= dim_size

        for pos in range(total_elements):
            current_out_idx: Index = np.zeros(len(out_shape), dtype=np.int32)
            to_index(pos, out_shape, current_out_idx)
            current_a_idx: Index = np.zeros(len(a_shape), dtype=np.int32)
            broadcast_index(current_out_idx, out_shape, a_shape, current_a_idx)
            current_b_idx: Index = np.zeros(len(b_shape), dtype=np.int32)
            broadcast_index(current_out_idx, out_shape, b_shape, current_b_idx)
            out_pos = index_to_position(current_out_idx, out_strides)
            a_pos = index_to_position(current_a_idx, a_strides)
            b_pos = index_to_position(current_b_idx, b_strides)
            out_storage[out_pos] = fn(
                a_storage[a_pos],
                b_storage[b_pos],
            )

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Reduces a tensor along a specified dimension using a binary function.

    Args:
    ----
        fn (Callable[[float, float], float]): The function to apply for reduction.

    Returns:
    -------
        Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]: The reduced tensor.

    """

    def _reduce(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        total_elements = 1
        for dim_size in out_shape:
            total_elements *= dim_size

        for pos in range(total_elements):
            current_out_idx: Index = np.zeros(len(out_shape), dtype=np.int32)
            to_index(pos, out_shape, current_out_idx)
            out_pos = index_to_position(current_out_idx, out_strides)
            current_a_idx: Index = current_out_idx.copy()

            for r in range(a_shape[reduce_dim]):
                current_a_idx[reduce_dim] = r
                a_pos = index_to_position(current_a_idx, a_strides)
                out_storage[out_pos] = fn(out_storage[out_pos], a_storage[a_pos])

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
