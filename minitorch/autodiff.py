from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol, Callable

# ## Task 1.1
# Central Difference calculation


def central_difference(
    f: Callable, *inputs: Any, target_arg: int = 0, delta: float = 1e-6
) -> Any:
    """Computes the central difference approximation for derivatives.

    Args:
    ----
        f (Callable): The function to differentiate.
        *inputs (Any): The input values to the function.
        target_arg (int, optional): The index of the argument to compute the derivative with respect to. Defaults to 0.
        delta (float, optional): The small change to apply for the central difference. Defaults to 1e-6.

    Returns:
    -------
        Any: The approximated derivative.

    """
    inputs_plus = list(inputs)
    inputs_minus = list(inputs)
    inputs_plus[target_arg] += delta
    inputs_minus[target_arg] -= delta
    f_plus = f(*inputs_plus)
    f_minus = f(*inputs_minus)
    difference = f_plus - f_minus
    return difference / (2 * delta)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add `x` to the derivative accumulated on this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Indicates if this variable is a leaf node (created by the user with no `last_fn`)."""
        ...

    def is_constant(self) -> bool:
        """Indicates if this variable is a constant (has no `derivative`)."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule for backpropagation."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Generates the topological order of the computation graph.

    Args:
    ----
        variable (Variable): The variable to start the topological sort from.

    Yields:
    ------
        Iterable[Variable]: Variables in topological order.

    """
    explored = set()
    sorted_list: List[Variable] = []

    def explore(node: Variable) -> None:
        if node.unique_id in explored or node.is_constant():
            return
        if not node.is_leaf():
            for ancestor in node.parents:
                if not ancestor.is_constant():
                    explore(ancestor)
        explored.add(node.unique_id)
        sorted_list.insert(0, node)

    explore(variable)
    return sorted_list


def backpropagate(variable: Variable, gradient: Any) -> None:
    """Performs backpropagation on the computation graph to compute derivatives for leaf nodes.

    Args:
    ----
        variable (Variable): The variable to backpropagate from.
        gradient (Any): The gradient to propagate.

    """
    sorted_vars = topological_sort(variable)
    gradients_map = {}
    gradients_map[variable.unique_id] = gradient

    for var in sorted_vars:
        current_grad = gradients_map[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(current_grad)
        else:
            for parent_var, local_grad in var.chain_rule(current_grad):
                if parent_var.is_constant():
                    continue
                if parent_var.unique_id not in gradients_map:
                    gradients_map[parent_var.unique_id] = 0.0
                gradients_map[parent_var.unique_id] += local_grad


@dataclass
class Context:
    """Context class used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *items: Any) -> None:
        """Stores the provided `items` for potential use during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = items

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrieves the saved tensors."""
        return self.saved_values
