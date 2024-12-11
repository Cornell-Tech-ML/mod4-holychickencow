from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """Modules form a hierarchical structure that store parameters and submodules.
    They serve as the foundational blocks for constructing neural network architectures.

    Attributes
    ----------
        _modules : Stores child modules indexed by their names.
        _parameters : Stores the module's parameters indexed by their names.
        training : Indicates whether the module is in training mode or evaluation mode.

    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """Retrieve the immediate child modules of this module."""
        child_modules: Dict[str, Module] = self.__dict__["_modules"]
        return list(child_modules.values())

    def train(self) -> None:
        """Activate training mode for this module and all its descendants."""
        self.training = True
        for child in self.modules():
            child.train()

    def eval(self) -> None:
        """Deactivate training mode (set to evaluation mode) for this module and all its descendants."""
        self.training = False
        for child in self.modules():
            child.eval()

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Gather all parameters of this module and its descendants, along with their hierarchical names.

        Returns
        -------
            A list of tuples containing the parameter names and their corresponding `Parameter` instances.

        """
        collected_params = dict(self._parameters)

        for mod_name, sub_module in self._modules.items():
            sub_params = sub_module.named_parameters()
            for param_name, param in sub_params:
                full_name = f"{mod_name}.{param_name}"
                collected_params[full_name] = param

        return list(collected_params.items())

    def parameters(self) -> Sequence[Parameter]:
        """Iterate over all parameters of this module and its descendants.

        Returns
        -------
            A list of `Parameter` instances.

        """
        return [param for _, param in self.named_parameters()]

    def add_parameter(self, name: str, value: Any) -> Parameter:
        """Explicitly add a parameter to this module. Useful for adding scalar or custom parameters.

        Args:
        ----
            name (str): The name of the parameter.
            value (Any): The value of the parameter.

        Returns:
        -------
            Parameter: The added parameter.

        """
        new_param = Parameter(value, name)
        self.__dict__["_parameters"][name] = new_param
        return new_param

    def __setattr__(self, key: str, value: Parameter) -> None:
        if isinstance(value, Parameter):
            self.__dict__["_parameters"][key] = value
        elif isinstance(value, Module):
            self.__dict__["_modules"][key] = value
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key: str) -> Any:
        params = self.__dict__.get("_parameters", {})
        modules = self.__dict__.get("_modules", {})
        if key in params:
            return params[key]
        if key in modules:
            return modules[key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Enable the module instance to be invoked as a callable, triggering the `forward` method.

        Args:
        ----
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.

        Returns:
        -------
            Any: The result of the forward method.

        """
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        def add_indentation(text: str, indent_spaces: int) -> str:
            lines = text.split("\n")
            if len(lines) == 1:
                return text
            first_line = lines.pop(0)
            indented_lines = [(" " * indent_spaces) + line for line in lines]
            return first_line + "\n" + "\n".join(indented_lines)

        module_descriptions = []

        for name, sub_module in self._modules.items():
            module_repr = repr(sub_module)
            module_repr = add_indentation(module_repr, 2)
            module_descriptions.append(f"({name}): {module_repr}")

        representation = f"{self.__class__.__name__}("

        if module_descriptions:
            # If there are child modules, list them with indentation
            representation += "\n  " + "\n  ".join(module_descriptions) + "\n"

        representation += ")"
        return representation


class Parameter:
    """A `Parameter` encapsulates a value within a `Module`, typically representing weights or biases.

    While designed to hold `Variable` instances, it can store any value type for testing purposes.
    """

    def __init__(self, value: Any, identifier: Optional[str] = None) -> None:
        self.value = value
        self.identifier = identifier
        if hasattr(value, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.identifier:
                self.value.name = self.identifier

    def update(self, new_value: Any) -> None:
        """Modify the stored parameter value.

        Args:
        ----
            new_value (Any): The new value to update the parameter with.

        """
        self.value = new_value
        if hasattr(new_value, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.identifier:
                self.value.name = self.identifier

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)
