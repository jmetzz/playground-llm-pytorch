import math
from collections.abc import Callable
from typing import Self

import numpy as np


class Operand:
    """
    Represents a numeric operand with support for basic arithmetic operations
    and automatic differentiation.

    Attributes:
        data (int | float): The numeric value of the operand.
        grad (float): The gradient associated with the operand, initialized to 0.0.
        label (str): Optional label for the operand, useful in computation graphs.
        src_operation (str): Source operation associated with this operand, if any.
        _src_operands (set[Operand]): Source operands used in the computation.
        _chain_backward_step (Callable): Function for backpropagating the gradient.
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
        Initializes an Operand with a given numeric value and optional metadata.

        Args:
            value (int | float): The numeric value to assign to this operand.
            label (str, optional): A label for this operand (e.g., variable name). Defaults to "".
            _src_operation (str, optional): The operation that produced this operand. Defaults to "".
            _src_operands (set[Operand] | None, optional): The operands involved in producing this value.
                Defaults to None.
            _backward (Callable, optional): The function defining the backward gradient step for this operand.
                Defaults to a no-op lambda.
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
        """Adds this operand to another operand.

        Args:
            other (Operand | int | float): The operand to add.

        Returns:
            Operand: A new Operand instance representing the sum of the two operands.
        """

        def _chain_backward_step():
            """Calculates the local derivative of the addition operation with respect to each operand
            and propagates the output gradient backward using the chain rule.

            In addition, each operand's contribution to the gradient is simply 1.0 (the local derivative for addition).
            The output gradient is distributed equally to each operand involved in the computation.

            Example:
                Given e = a + b, the derivatives with respect to each operand are:
                    de/da = 1.0
                    de/db = 1.0

                The local derivative is calculated as
                    de/da = (f(x + h) - f(x))/h
                            = ((a+b + h) - (a+b))/h
                            = (h + (a+b) - (a+b))/h
                            = h/h
                            = 1.0


                Applying the chain rule:
                    a.grad += 1.0 * output.grad
                    b.grad += 1.0 * output.grad

            This means that each operand's gradient is incremented by the product of its local derivative
            (1.0 for addition) and the output gradient.

            Note:
                The `+=` operation ensures the function is cumulative, covering multivariate cases and situations
                where a node is reused in the graph.
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
        Adds a scalar to the operand's value in a right-side operation.

        Args:
            other (int | float): The scalar to add.

        Returns:
            Operand: A new Operand instance representing the sum of the scalar and this operand.
        """
        return self + other

    def __neg__(self) -> Self:
        """
        Negates the operand's value.

        Returns:
            Operand: A new Operand instance representing the negation of the original operand.
        """
        return self * -1

    def __sub__(self, other) -> Self:
        """
        Subtracts another operand from this operand.

        Args:
            other (Operand | int | float): The operand to subtract.

        Returns:
            Operand: A new Operand instance representing the difference between this operand and the other operand.
        """
        return self + (-other)

    def __rsub__(self, other) -> Self:
        """
        Subtracts the operand's value from a scalar in a right-side operation.

        Args:
            other (int | float): The scalar from which to subtract the operand's value.

        Returns:
            Operand: A new Operand instance representing the result of subtracting this operand from the scalar.
        """
        return other + (-self)

    def __mul__(self, other) -> Self:
        """
        Multiplies this operand by another operand.

        Args:
            other (Operand | int | float): The operand to multiply with.

        Returns:
            Operand: A new Operand instance representing the product of the two operands.
        """

        def _chain_backward_step():
            """
            Calculates the local derivatives of the multiplication operation with respect to each operand
            and propagates the output gradient backward using the chain rule.

            For multiplication, the local derivatives with respect to each operand are:
                - de/da = b
                - de/db = a

            Example:
                Given e = a * b, the derivatives with respect to each operand are:
                    de/da = b
                    de/db = a

                Applying the chain rule:
                    a.grad += b * output.grad
                    b.grad += a * output.grad

                This updates each operand's gradient by multiplying the output gradient by the
                local derivative for that operand.

            Note:
                The `+=` operator is used to ensure cumulative updates to cover multivariate cases
                and scenarios where a node is reused in the computation graph.
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
        Multiplies a scalar with the operand's value in a right-side operation.

        Args:
            other (int | float): The scalar to multiply with.

        Returns:
            Operand: A new Operand instance representing the product of the scalar and this operand's value.
        """
        return self * other

    def __truediv__(self, other) -> Self:
        """
        Divides this operand by another operand.

        Uses the formula:

            a / b = a * (1 / b) = a * b^(-1)

        Args:
            other (Operand | int | float): The operand to divide by.

        Returns:
            Operand: A new Operand instance representing the division of this operand by the other operand.
        """
        return self * other**-1

    def __rtruediv__(self, other) -> Self:
        """
        Divides a scalar by the operand's value in a right-side operation.

        Args:
            other (int | float): The scalar to divide by the operand's value.

        Returns:
            Operand: A new Operand instance representing the result of dividing the scalar by this operand's value.
        """
        return other * self**-1

    def __pow__(self, other: int | float) -> Self:
        """
        Raises the operand's value to the power of a given exponent.

        Args:
            other (int | float): The exponent.

        Raises:
            ValueError: If the exponent is not an integer or float.

        Returns:
            Operand: A new Operand instance with the value raised to the specified power.
        """

        def _chain_backward_step():
            """
            Calculates the local derivative of the power operation with respect to the operand
            and propagates the output gradient backward using the chain rule.

            For an exponentiation operation of the form e = a^n, where `a` is the base operand and
            `n` is a constant exponent, the local derivative with respect to `a` is calculated as:
                de/da = n * a^(n - 1)

            Example:
                Given e = a^n, the derivative with respect to `a` is:
                    de/da = n * a^(n - 1)

                Applying the chain rule:
                    a.grad += n * a^(n - 1) * output.grad

                This updates the gradient of `a` by multiplying the output gradient by the local
                derivative with respect to `a`.

            Note:
                The `+=` operator is used to accumulate gradients, supporting multivariate cases
                and scenarios where nodes are reused in the computation graph.
            """
            self.grad += other * self.data ** (other - 1) * output.grad

        if not isinstance(other, int | float):
            raise ValueError("pow only supports int or float powers")

        output = Operand(
            self.data**other, _src_operation=f"**{other}", _src_operands=(self,), _backward=_chain_backward_step
        )
        return output  # noqa: RET504

    def tanh(self) -> Self:
        """
        Computes the hyperbolic tangent of the operand's value.

        The tanh function is defined as:
            tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)

        The derivative is:
            d/dx tanh(x) = 1 - tanh(x)^2

        Returns:
            Operand: A new Operand instance representing the tanh of the original value.
        """

        def _chain_backward_step():
            """
            Computes the local derivative of the tanh operation with respect to the operand
            and propagates the output gradient backward using the chain rule.

            For the hyperbolic tangent function, tanh(x), the local derivative with respect to `x` is:
                d(tanh(x))/dx = 1 - tanh(x)^2

            Example:
                Given e = tanh(a), the local derivative with respect to `a` is:
                    de/da = 1 - tanh(a)^2

                Applying the chain rule:
                    a.grad += (1 - tanh(a)^2) * output.grad

                This updates the gradient of `a` by multiplying the output gradient by the
                local derivative of tanh with respect to `a`.

            Note:
                The `+=` operator is used to ensure cumulative updates for multivariate cases
                and scenarios where nodes are reused in the computation graph.
            """
            self.grad += (1 - new_value**2) * output.grad

        value = self.data

        # new_value should be (math.exp(2 * value) - 1) / (math.exp(2 * value) + 1)
        # but this implementation is not stable and causes division by zero every now and then.
        # Thus, using numpy's implementation here to solve this issue
        new_value = np.tanh(value).item()
        output = Operand(new_value, _src_operation="tanh", _src_operands=(self,), _backward=_chain_backward_step)
        return output  # noqa: RET504

    def relu(self):
        """
        Computes the Rectified Linear Unit (ReLU) activation of the operand's value.

        The ReLU function is defined as:
            relu(x) = max(0, x)

        This function also defines the derivative of the ReLU function:
            d/dx relu(x) = 1 if x > 0 else 0

        At x = 0, the derivative is typically set to 0 in implementations.

        Returns:
            Operand: A new Operand instance containing the ReLU activation of the original value.

        Example:
            >>> operand = Operand(data=-3)
            >>> relu_output = operand.relu()
            >>> print(relu_output.data)  # Outputs: 0
        """

        def _chain_backward_step():
            self.grad += (output.data > 0) * output.grad

        output = Operand(
            max(self.data, 0),
            _src_operation="ReLU",
            _src_operands=(self,),
            _backward=_chain_backward_step,
        )
        return output  # noqa: RET504

    def exp(self) -> Self:
        """
        Computes the exponential of the operand's value.

        The exponential function is defined as:
            e^x

        The derivative is:
            d/dx e^x = e^x

        Returns:
            Operand: A new Operand instance representing e^x of the original value.
        """

        def _chain_backward_step():
            """
            Computes the local derivative of the exponential operation with respect to the operand
            and propagates the output gradient backward using the chain rule.

            For the exponential function, exp(x), the local derivative with respect to `x` is:
                d(exp(x))/dx = exp(x)

            Example:
                Given e = exp(a), the local derivative with respect to `a` is:
                    de/da = exp(a)

                Applying the chain rule:
                    a.grad += exp(a) * output.grad

                This updates the gradient of `a` by multiplying the output gradient by the local
                derivative of exp with respect to `a`.

            Note:
                The `+=` operator should be used if cumulative updates are required in cases where
                nodes are reused in the computation graph. Here, `=` is sufficient for this operation.
            """
            self.grad += output.data * output.grad

        x = self.data
        output = Operand(math.exp(x), _src_operation="exp", _src_operands=(self,), _backward=_chain_backward_step)
        return output  # noqa: RET504

    def sigmoid(self) -> Self:
        def _chain_backward_step():
            # Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
            _local = output.data * (1 - output.data)
            self.grad += _local * output.grad

        x = self.data
        output = Operand(
            1 / (1 + (math.exp(-x))),
            _src_operation="Sigmoid",
            _src_operands=(self,),
            _backward=_chain_backward_step,
        )
        return output  # noqa: RET504

    def trace(self) -> tuple[set[Self], set[tuple[Self]]]:
        """
        Builds a trace of the computation graph from the current operand, identifying
        all nodes and edges in topological order.

        Returns:
            tuple[set[Operand], set[tuple[Operand]]]: A tuple containing:
                - nodes: A set of Operand nodes in the computation graph.
                - edges: A set of edges representing dependencies between operands.
        """

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
        """
        Performs the backward step for automatic differentiation, calculating gradients
        for each operand in the computation graph.

        The function traverses the computation graph in topological order to apply
        the chain rule for gradient propagation.
        """

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
