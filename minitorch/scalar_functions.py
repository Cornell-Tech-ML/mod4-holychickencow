from __future__ import annotations

from typing import TYPE_CHECKING, Any  # Add Any to the import

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @staticmethod
    def forward(ctx: Context, *args: Any) -> float:
        """Forward pass to be implemented by subclasses."""
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, d_output: Any) -> Any:
        """Backward pass to be implemented by subclasses."""
        raise NotImplementedError

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the given values.

        Args:
        ----
            *vals (ScalarLike): The input values.

        Returns:
        -------
            Scalar: The result of applying the function.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls.forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass of the addition operation.

        Args:
        ----
            ctx (Context): The context for saving values for backward pass.
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: The result of a + b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass of the addition operation.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient flowing back from the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients with respect to the inputs.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the log operation.

        Args:
        ----
            ctx (Context): The context for saving values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The result of log(a).

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the log operation.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient flowing back from the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.
class Mul(ScalarFunction):
    """Multiplication function f(x, y) = x * y"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Compute the forward pass of the multiplication operation.

        Args:
        ----
            ctx (Context): The context for saving values for backward pass.
            x (float): The first input.
            y (float): The second input.

        Returns:
        -------
            float: The result of x * y.

        """
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass of the multiplication operation.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient flowing back from the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to the inputs.

        """
        x, y = ctx.saved_values
        return d_output * y, d_output * x


class Inv(ScalarFunction):
    """Inverse function f(x) = 1 / x"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Compute the forward pass of the inverse operation.

        Args:
        ----
            ctx (Context): The context for saving values for backward pass.
            x (float): The input value.

        Returns:
        -------
            float: The result of 1 / x.

        """
        ctx.save_for_backward(x)
        return 1.0 / x

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the inverse operation.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient flowing back from the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (x,) = ctx.saved_values
        return d_output * (-1.0 / (x**2))


class Neg(ScalarFunction):
    """Negation function f(x) = -x"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Compute the forward pass of the negation operation.

        Args:
        ----
            ctx (Context): The context for saving values for backward pass.
            x (float): The input value.

        Returns:
        -------
            float: The result of -x.

        """
        return -x

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the negation operation.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient flowing back from the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function f(x) = 1 / (1 + exp(-x))"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Compute the forward pass of the sigmoid operation.

        Args:
        ----
            ctx (Context): The context for saving values for backward pass.
            x (float): The input value.

        Returns:
        -------
            float: The result of the sigmoid function.

        """
        sigmoid_x = operators.sigmoid(x)
        ctx.save_for_backward(sigmoid_x)
        return sigmoid_x

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the sigmoid operation.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient flowing back from the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (sigmoid_x,) = ctx.saved_values
        return d_output * sigmoid_x * (1 - sigmoid_x)


class ReLU(ScalarFunction):
    """ReLU function f(x) = x if x > 0 else 0"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Compute the forward pass of the ReLU operation.

        Args:
        ----
            ctx (Context): The context for saving values for backward pass.
            x (float): The input value.

        Returns:
        -------
            float: The result of the ReLU function.

        """
        ctx.save_for_backward(x)
        return x if x > 0 else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the ReLU operation.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient flowing back from the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (x,) = ctx.saved_values
        return d_output if x > 0 else 0.0


class Exp(ScalarFunction):
    """Exponential function f(x) = exp(x)"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Compute the forward pass of the exponential operation.

        Args:
        ----
            ctx (Context): The context for saving values for backward pass.
            x (float): The input value.

        Returns:
        -------
            float: The result of the exponential function.

        """
        y = operators.exp(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the exponential operation.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient flowing back from the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (y,) = ctx.saved_values
        return d_output * y


class LT(ScalarFunction):
    """Less than function f(x, y) = 1.0 if x < y else 0.0"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Compute the forward pass of the less than operation.

        Args:
        ----
            ctx (Context): The context for saving values for backward pass.
            x (float): The first input.
            y (float): The second input.

        Returns:
        -------
            float: The result of the less than comparison.

        """
        return 1.0 if x < y else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass of the less than operation.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient flowing back from the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to the inputs.

        """
        # Derivative of comparison functions is zero
        return 0.0, 0.0


class GT(ScalarFunction):
    """Greater than function f(x, y) = 1.0 if x > y else 0.0"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Compute the forward pass of the greater than operation.

        Args:
        ----
            ctx (Context): The context for saving values for backward pass.
            x (float): The first input.
            y (float): The second input.

        Returns:
        -------
            float: The result of the greater than comparison.

        """
        return 1.0 if x > y else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass of the greater than operation.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient flowing back from the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to the inputs.

        """
        # Derivative of comparison functions is zero
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function f(x, y) = 1.0 if x == y else 0.0"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Compute the forward pass of the equality operation.

        Args:
        ----
            ctx (Context): The context for saving values for backward pass.
            x (float): The first input.
            y (float): The second input.

        Returns:
        -------
            float: The result of the equality comparison.

        """
        return 1.0 if x == y else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass of the equality operation.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient flowing back from the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to the inputs.

        """
        # Derivative of comparison functions is zero
        return 0.0, 0.0


# Additional functions needed for scalar.py


class Sub(ScalarFunction):
    """Subtraction function f(x, y) = x - y"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Compute the forward pass of the subtraction operation.

        Args:
        ----
            ctx (Context): The context for saving values for backward pass.
            x (float): The first input.
            y (float): The second input.

        Returns:
        -------
            float: The result of x - y.

        """
        return x - y

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass of the subtraction operation.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient flowing back from the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to the inputs.

        """
        return d_output * 1.0, d_output * -1.0
