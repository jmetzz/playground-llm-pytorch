import math

import pytest

from micrograd.engine import Operand


@pytest.mark.parametrize(
    ("scalar_1", "scalar_2", "expected"), [(0, 0, 0), (0, 1, 1), (1, 0, 1), (0, -1, -1), (-1, 0, -1)]
)
def test_operation_add(scalar_1, scalar_2, expected):
    assert (Operand(scalar_1) + Operand(scalar_2)).data == expected
    assert (scalar_1 + Operand(scalar_2)).data == expected
    assert (Operand(scalar_1) + scalar_2).data == expected


@pytest.mark.parametrize(
    ("scalar_1", "scalar_2", "expected"), [(0, 0, 0), (0, 1, -1), (1, 0, 1), (0, -1, 1), (-1, 0, -1)]
)
def test_operation_sub(scalar_1, scalar_2, expected):
    assert (Operand(scalar_1) - Operand(scalar_2)).data == expected
    assert (scalar_1 - Operand(scalar_2)).data == expected
    assert (Operand(scalar_1) - scalar_2).data == expected


@pytest.mark.parametrize(("scalar", "expected"), [(0, 0), (1, -1), (-1, 1), (3, -3)])
def test_operation_neg(scalar, expected):
    actual = -Operand(scalar)
    assert actual.data == expected


@pytest.mark.parametrize(
    ("scalar_1", "scalar_2", "expected"),
    [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, -1, -1), (-1, 1, -1), (2, 1, 2), (1, 2, 2), (2, 2, 4), (3, 2, 6), (2, 3, 6)],
)
def test_operation_mul(scalar_1, scalar_2, expected):
    assert (Operand(scalar_1) * Operand(scalar_2)).data == expected
    assert (scalar_1 * Operand(scalar_2)).data == expected
    assert (Operand(scalar_1) * scalar_2).data == expected


@pytest.mark.parametrize(
    ("scalar_1", "scalar_2", "expected"),
    [
        (0, 1, 0),
        (2, 1, 2),
        (1, 2, 0.5),
        (2, 2, 1),
        (9, 3, 3),
        (62.5, 5, 12.5),
    ],
)
def test_operation_div(scalar_1, scalar_2, expected):
    assert (Operand(scalar_1) / Operand(scalar_2)).data == expected
    assert (scalar_1 / Operand(scalar_2)).data == expected
    assert (Operand(scalar_1) / scalar_2).data == expected


@pytest.mark.parametrize("scalar", [0, 1, 2, 3, 5, 2.5])
def test_operation_exp(scalar):
    operand = Operand(scalar)
    assert operand.exp().data == math.exp(scalar)


@pytest.mark.parametrize(
    ("scalar", "power", "expected"),
    [
        (0, 1, 0),
        (2, 1, 2),
        (1, 2, 1),
        (2, 2, 4),
        (5, 2, 25),
        (3, 3, 27),
        (2, -1, 0.5),
        (4, -1, 0.25),
        (10, -1, 0.1),
        (2, -2, 0.25),
        (3, -2, 1 / 9),
        (5, -3, 1 / 125),
    ],
)
def test_operation_pow(scalar, power, expected):
    assert (Operand(scalar) ** power).data == expected


@pytest.mark.parametrize(
    ("scalar", "expected"),
    [
        (-3, -0.9950547536867306),
        (-2, -0.9640275800758168),
        (-1, -0.7615941559557649),
        (0, 0.0),
        (1, 0.7615941559557649),
        (2, 0.9640275800758169),
        (3, 0.9950547536867305),
    ],
)
def test_operation_tanh(scalar, expected):
    operand = Operand(scalar)
    assert operand.tanh().data == expected
