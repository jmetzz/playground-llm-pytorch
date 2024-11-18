import math
from collections.abc import Callable
from typing import Self


class Operand:
    """
    Represents a numeric operand with support for basic arithmetic operations
    and automatic differentiation.

    Attributes:
        data (int | float): The numeric value of the operand.
        grad (float): The gradient associated with the operand, initialized to 0.0.
    """

    def __init__(
        self,
        value: int | float,
        label: str = "",
        _src_operation: str = "",
        _src_operands: set | None = None,
        _backward: Callable = lambda: None,
    ) -> None:
        """
        Initializes an Operand with a given numeric value.

        Args:
            value (int | float): The numeric value to assign to this operand.
        """
        self.data = value
        self.grad = 0.0
        # metadata for visualization of the computation graph
        self.label = label
        self.src_operation = _src_operation
        self._src_operands = set(_src_operands) if _src_operands else set()  # enforce set
        # the backward derivative function
        self._chain_backward_step = _backward

    def __add__(self, other) -> Self:
        """
        Adds this operand to another operand.

        Args:
            other (Operand): The operand to add.

        Returns:
            Operand: A new operand representing the sum of the two operands.
        """

        def _chain_backward_step():
            """The derivative of the + operation.

            Calculates the local derivative, which is 1.0 for addition, and
            propagates the output gradient back using the chain rule.

            This means, it only "propagates/routes" backwards the output for addition operation.
                gradient +=  local_derivative * output_gradient

            Example:
            given e = a + b, the local derivative is calculated as
                    de/da = (f(x + h) - f(x))/h
                          = ((a+b + h) - (a+b))/h
                          = (h + (a+b) - (a+b))/h
                          = h/h
                          = 1.0

                likewise,
                    de/db = 1.0

                Then, applying the chain rule:
                    gradient_a += 1.0 * output_gradient
                    gradient_b += 1.0 * output_gradient

            Alternatively:
                self.grad += 1.0 * output.grad
                other.grad += 1.0 * output.grad

            The multiplication between these two elements is actually the chain rule.

            Note: This should be a cumulative function to cover multivariate cases
            and when a node is reused in the graph. Thus, the += assignment.
            """
            self.grad += output.grad
            other.grad += output.grad

        other = other if isinstance(other, Operand) else Operand(other)
        output = Operand(
            self.data + other.data, _src_operation="+", _src_operands=(self, other), _backward=_chain_backward_step
        )
        return output  # noqa: RET504

    def __radd__(self, other: int | float) -> Self:
        """
        Adds a scalar to the operand's value.

        Args:
            other (int | float): The scalar to add.

        Returns:
            Operand: A new operand representing the sum of the scalar and the operand's value.
        """
        return self + other

    def __neg__(self) -> Self:
        return self * -1

    def __sub__(self, other) -> Self:
        """
        Subtracts another operand from this operand.

        Args:
            other (Operand): The operand to subtract.

        Returns:
            Operand: A new operand representing the difference of the two operands.
        """
        return self + (-other)

    def __rsub__(self, other) -> Self:
        """
        Subtracts the operand's value from a scalar.

        Args:
            other (int | float): The scalar to subtract from.

        Returns:
            Operand: A new operand representing the result of subtracting the operand's value from the scalar.
        """
        return other + (-self)

    def __mul__(self, other) -> Self:
        """
        Multiplies this operand by another operand.

        Args:
            other (Operand): The operand to multiply with.

        Returns:
            Operand: A new operand representing the product of the two operands.
        """

        def _chain_backward_step():
            """
            Calculates the local derivative wrt multiplication of operands and
            propagates the output gradient back using the chain rule.

            given e = a * b, the local derivative is calculated as
                de / da = b

            likewise,
                de / db = a

            Then, applying the chain rule:
                gradient_a = b * output_gradient
                gradient_b = a * output_gradient

            Note: This should be a cumulative function to cover multivariate cases
            and when a node is reused in the graph. Thus, the += assignment.
            """
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        other = other if isinstance(other, Operand) else Operand(other)
        output = Operand(
            self.data * other.data, _src_operation="*", _src_operands=(self, other), _backward=_chain_backward_step
        )
        return output  # noqa: RET504

    def __rmul__(self, other) -> Self:
        """
        Multiplies a scalar with the operand's value.

        Args:
            other (int | float): The scalar to multiply with.

        Returns:
            Operand: A new operand representing the product of the scalar and the operand's value.
        """
        return self * other

    def __truediv__(self, other) -> Self:
        """
        Divides this operand by another operand.

        Uses the formula:

        .. math::
            \\frac{a}{b} = a \\times \\frac{1}{b} = a \\times b^{-1}

        Args:
            other (Operand): The operand to divide by.

        Returns:
            Operand: A new operand representing the division of the two operands.
        """
        return self * other**-1

    def __rtruediv__(self, other) -> Self:
        """
        Divides a scalar by the operand's value.

        Args:
            other (int | float): The scalar to divide by the operand's value.

        Returns:
            Operand: A new operand representing the result of dividing the scalar by the operand's value.
        """
        return other * self**-1

    def __pow__(self, other: int | float) -> Self:
        """
        Raises the operand's value to the power of a given exponent.

        Args:
            exponent (int | float): The exponent to which the operand's value will be raised.

        Raises:
            ValueError: If the exponent is not of type int or float.

        Returns:
            Operand: A new operand representing the result of raising the original operand's value
            to the given exponent.
        """

        def _chain_backward_step():
            self.grad += other * self.data ** (other - 1) * output.grad

        if not isinstance(other, int | float):
            raise ValueError("pow only supports int or float powers")

        output = Operand(
            self.data**other, _src_operation=f"**{other}", _src_operands=(self,), _backward=_chain_backward_step
        )
        return output  # noqa: RET504

    def tanh(self) -> Self:
        """
        Computes the hyperbolic tangent of the input.

        The tanh function is defined as:

        .. math::
            \tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}

        where :math:`e` is the base of the natural logarithm.

        The derivative of the tanh function is given by:

        .. math::
            \frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)


        Returns:
            Operand: A new operand representing the hyperbolic tangent of self.
        """

        def _chain_backward_step():
            """Computes the gradient of the tanh operation.

            The derivative is computed as:
            .. math::
                \frac{d}{dx}\tanh(x) = 1 - \tanh(x)^2
            """
            self.grad += (1 - new_value**2) * output.grad

        value = self.data
        new_value = (math.exp(2 * value) - 1) / (math.exp(2 * value) + 1)
        output = Operand(new_value, _src_operation="tanh", _src_operands=(self,), _backward=_chain_backward_step)
        return output  # noqa: RET504

    def exp(self) -> Self:
        """
        Computes the exponential of the operand's value.

        The exponential function is defined as:

        .. math::
            e^x

        where :math:`e` is the base of the natural logarithm and :math:`x` is the operand's value.

        The derivative of the exponential function is given by:

        .. math::
            \frac{d}{dx} e^x = e^x

        Returns:
            Operand: A new operand representing the exponential of the original operand's value.
        """

        def _chain_backward_step():
            """Computes the gradient of the exponential operation.

            The derivative is computed as:
            .. math::
                \frac{dy}{dx} = e^x
            where :math:`y = e^x`.
            """
            # self.grad += output.data * output.grad
            self.grad = output.data * output.grad

        x = self.data
        output = Operand(math.exp(x), _src_operation="exp", _src_operands=(self,), _backward=_chain_backward_step)
        return output  # noqa: RET504

    def trace(self) -> tuple[set[Self], set[tuple[Self]]]:
        nodes, edges = set(), set()

        def _dfs_build(node):
            """Topological order dependency graph"""
            if node not in nodes:
                nodes.add(node)
                for ancestor in node._src_operands:  # noqa: SLF001
                    edges.add((ancestor, node))
                    _dfs_build(ancestor)

        _dfs_build(self)
        return nodes, edges

    def __repr__(self) -> str:
        """
        Returns a string representation of the Operand object.

        Returns:
            str: The string representation of the operand's value.
        """
        return f"Operand(data={self.data})"

    def backward(self) -> None:
        """The backward step on the computation graph"""

        topological_order = []
        visited = set()

        def _dfs_build_topological_order(node):
            if node not in visited:
                visited.add(node)
                for ancestor in node._src_operands:  # noqa: SLF001
                    _dfs_build_topological_order(ancestor)
                topological_order.append(node)

        _dfs_build_topological_order(self)

        # Apply the chain rule to get its gradient
        self.grad = 1.0
        for operand in reversed(topological_order):
            operand._chain_backward_step()  # noqa: SLF001
