"""Collection of the core mathematical operators used throughout the code base."""

import math
import operator

# ## Task 0.1
from typing import Callable, List, Sequence

#
# Implementation of a prelude of elementary functions.
# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y


def mul(x: float, y: float) -> float:
    """Multiply two floats."""
    return x * y  # Ensure this always returns a float


def id(x: float) -> float:
    """Identity function."""
    return x  # Ensure this always returns a float


def neg(x: float) -> float:
    """Negates the input."""
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if x is less than y."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if x is equal to y."""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the maximum of x and y."""
    return x if x > y else y


def is_close(x: float, y: float, tol: float = 1e-2) -> bool:
    """Checks if x and y are close within a tolerance."""
    return abs(x - y) < tol


def sigmoid(x: float) -> float:
    """Computes the sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Computes the ReLU function."""
    if x > 0.0:
        return x
    else:
        return 0.0


def log(x: float) -> float:
    """Computes the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Computes the exponential function."""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Computes the gradient of the logarithm function."""
    return y / x


def inv(x: float) -> float:
    """Computes the inverse of x."""
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    """Computes the gradient of the inverse function."""
    return -y / (x**2)


def relu_back(x: float, y: float) -> float:
    """Computes the gradient of the ReLU function."""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float], lst: List[float]) -> List[float]:
    """Applies a function to each element in a list."""
    return [fn(x) for x in lst]


def zipWith(
    fn: Callable[[float, float], float], lst1: List[float], lst2: List[float]
) -> List[float]:
    """Applies a function to pairs of elements from two lists."""
    return [fn(x, y) for x, y in zip(lst1, lst2)]


def reduce(
    fn: Callable[[float, float], float], lst: Sequence[float], start: float
) -> float:
    """Reduces a list to a single value by applying a binary function cumulatively."""
    result = start
    for x in lst:
        result = fn(result, x)
    return result


def negList(lst: List[float]) -> List[float]:
    """Negates each element in a list."""
    return map(neg, lst)


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Adds two lists element-wise."""
    return zipWith(add, lst1, lst2)


def sum(lst: List[float]) -> float:
    """Computes the sum of a list."""
    return reduce(add, lst, 0.0)


def prod(lst: Sequence[float]) -> float:
    """Product of a list of numbers."""
    return reduce(operator.mul, lst, 1.0)
