import random
from collections.abc import Sequence
from enum import StrEnum

from micrograd.engine import Operand


class ActivationChoice(StrEnum):
    RELU = "ReLU"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LINEAR = "linear"


class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list:  # noqa: PLR6301
        return []


class Neuron(Module):
    def __init__(self, num_inputs: int, activation: ActivationChoice = "ReLU") -> None:
        self.weights = [Operand(random.uniform(-1, 1)) for _ in range(num_inputs)]  # noqa: S311
        self.bias = Operand(random.uniform(0, 1))  # noqa: S311
        self.activation = activation

    def __call__(self, inputs: Sequence[Operand | int | float]) -> Operand:
        # calculate the (x*w + b) expression for this neuron
        value = sum((xi * wi for xi, wi in zip(inputs, self.weights, strict=False)), start=self.bias)
        # return the result of the activation function
        match self.activation:
            case ActivationChoice.RELU:
                return value.relu()
            case ActivationChoice.TANH:
                return value.tanh()
            case ActivationChoice.SIGMOID:
                return value.sigmoid()
            case _:
                return value

    def parameters(self) -> list:
        return [*self.weights, self.bias]

    def __repr__(self):
        return f"{self.activation}Neuron({len(self.weights)})"


class Layer(Module):
    def __init__(self, num_inputs: int, num_outputs: int, **kwargs) -> None:
        self.neurons = [Neuron(num_inputs, **kwargs) for _ in range(num_outputs)]

    def __call__(self, inputs: Sequence[Operand | int | float]) -> Sequence[Operand]:
        output = [neuron(inputs) for neuron in self.neurons]
        return output[0] if len(output) == 1 else output

    def parameters(self) -> list:
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, num_inputs: int, layers_num_outputs: list[int]) -> None:
        sizes = [num_inputs, *layers_num_outputs]
        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                activation=ActivationChoice.TANH if i != len(layers_num_outputs) - 1 else ActivationChoice.LINEAR,
            )
            for i in range(len(layers_num_outputs))
        ]

    def __call__(self, inputs: Sequence[Operand | int | float]) -> Sequence[Operand]:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(
        self,
    ) -> list:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
