from typing import Self


class Operand:
    """
    Represents a numeric operand with support for basic arithmetic operations
    and automatic differentiation.

    Attributes:
        data (int | float): The numeric value of the operand.
        grad (float): The gradient associated with the operand, initialized to 0.0.
    """

    def __init__(self, value: int | float) -> None:
        """
        Initializes an Operand with a given numeric value.

        Args:
            value (int | float): The numeric value to assign to this operand.
        """
        self.data = value
        self.grad = 0.0

    def __add__(self, other) -> Self:
        """
        Adds this operand to another operand.

        Args:
            other (Operand): The operand to add.

        Returns:
            Operand: A new operand representing the sum of the two operands.
        """
        return Operand(self.data + other.data)

    def __sub__(self, other) -> Self:
        """
        Subtracts another operand from this operand.

        Args:
            other (Operand): The operand to subtract.

        Returns:
            Operand: A new operand representing the difference of the two operands.
        """
        return Operand(self.data - other.data)

    def __mul__(self, other) -> Self:
        """
        Multiplies this operand by another operand.

        Args:
            other (Operand): The operand to multiply with.

        Returns:
            Operand: A new operand representing the product of the two operands.
        """
        return Operand(self.data * other.data)

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
        return Operand(self.data * other.data**-1)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Operand object.

        Returns:
            str: The string representation of the operand's value.
        """
        return f"Operand(data={self.data})"


if __name__ == "__main__":
    a = Operand(2.0)
    b = Operand(-3.0)

    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b)
