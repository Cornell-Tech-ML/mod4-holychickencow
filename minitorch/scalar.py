from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    Add,
    Sub,
    Mul,
    Inv,
    Neg,
    Exp,
    Log,
    Sigmoid,
    ReLU,
    LT,
    GT,
    EQ,
    ScalarFunction,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __bool__(self) -> bool:
        return bool(self.data)

    # Overridden mathematical operators

    def __add__(self, other: ScalarLike) -> Scalar:
        return Add.apply(self, other)

    def __radd__(self, other: ScalarLike) -> Scalar:
        return self + other

    def __sub__(self, other: ScalarLike) -> Scalar:
        return Sub.apply(self, other)

    def __rsub__(self, other: ScalarLike) -> Scalar:
        return Sub.apply(other, self)

    def __mul__(self, other: ScalarLike) -> Scalar:
        return Mul.apply(self, other)

    def __rmul__(self, other: ScalarLike) -> Scalar:
        return self * other

    def __truediv__(self, other: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(other))

    def __rtruediv__(self, other: ScalarLike) -> Scalar:
        return Mul.apply(other, Inv.apply(self))

    def __neg__(self) -> Scalar:
        return Neg.apply(self)

    def __lt__(self, other: ScalarLike) -> Scalar:
        return LT.apply(self, other)

    def __gt__(self, other: ScalarLike) -> Scalar:
        return GT.apply(self, other)

    def __eq__(self, other: ScalarLike) -> Scalar:
        return EQ.apply(self, other)

    # Mathematical functions

    def log(self) -> Scalar:
        """Compute the natural logarithm of this scalar.

        Returns
        -------
            Scalar: A new scalar representing the result of log(self).

        """
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Compute the exponential of this scalar.

        Returns
        -------
            Scalar: A new scalar representing the result of exp(self).

        """
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Compute the sigmoid of this scalar.

        Returns
        -------
            Scalar: A new scalar representing the result of sigmoid(self).

        """
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Compute the ReLU (Rectified Linear Unit) of this scalar.

        Returns
        -------
            Scalar: A new scalar representing the result of relu(self).

        """
        return ReLU.apply(self)

    # Variable elements for backpropagation

    def accumulate_derivative(self, x: Any) -> None:
        """Add `x` to the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: Value to be accumulated.

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        """True if this variable was created by the user (no `last_fn`)."""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable is a constant (no history)."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Return the inputs that created this variable."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to compute gradients."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        # Get the local derivatives by calling backward.
        local_derivatives = h.last_fn.backward(h.ctx, d_output)

        # Ensure that local_derivatives is a tuple
        if not isinstance(local_derivatives, tuple):
            local_derivatives = (local_derivatives,)

        # Pair each derivative with its corresponding input.
        # Filter out constants (inputs that are constants).
        result = []
        for inp, deriv in zip(h.inputs, local_derivatives):
            if not inp.is_constant():
                result.append((inp, deriv))

        return result

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output: Starting derivative to backpropagate through the model
                      (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    def backpropagate(self, variable: Variable) -> None:
        """Backpropagate through the scalar."""
        if isinstance(variable, Scalar):
            variable.backward()


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Check that autodiff works correctly on a Python function.

    Args:
    ----
        f (Any): The function to check.
        *scalars (Scalar): The input scalars to the function.

    Raises:
    ------
        AssertionError: If the computed derivative is incorrect.

    """
    # Perform the forward and backward pass with the original scalars
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""

    # Extract the float values from the scalars
    scalar_vals = [s.data for s in scalars]

    # Define the wrapper function
    def f_raw(*vals: float) -> float:
        # Wrap float inputs into Scalars
        scalars_new = [Scalar(v) for v in vals]
        # Call the original function
        result = f(*scalars_new)
        # Return the float data
        return result.data

    for i, x in enumerate(scalars):
        # Compute the central difference using f_raw
        check = central_difference(f_raw, *scalar_vals)
        print(str([s.data for s in scalars]), x.derivative, i, check)
        assert x.derivative is not None

        # Compare the derivative from autodiff with the central difference
        np.testing.assert_allclose(
            x.derivative,
            check,
            rtol=1e-2,
            atol=1e-2,
            err_msg=err_msg % (str([s.data for s in scalars]), x.derivative, i, check),
        )
