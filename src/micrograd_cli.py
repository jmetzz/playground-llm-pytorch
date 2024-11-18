import logging

import colorama
import typer
from colorama import Fore
from environs import Env

from micrograd.engine import Operand
from micrograd.nn import MLP, Layer, Neuron
from micrograd.visualization import build_computation_graph, plot_computation_graph

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

env = Env()
env.read_env()  # read .env file, if it exists

# Get log level from .env and convert to logging level
log_level_name = env("LOG_LEVEL", default="INFO").upper()  # Default to INFO
log_level = logging.getLevelName(log_level_name.upper())
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

colorama.init(autoreset=True)

app = typer.Typer()


@app.command()
def neuron_expr():
    # inputs
    x1 = Operand(2.0, label="x1")
    x2 = Operand(0.0, label="x2")

    # weights
    w1 = Operand(-3.0, label="w1")
    w2 = Operand(1.0, label="w2")

    # bias
    b = Operand(6.881373587, label="b")

    # the neuron expression: (x1 * w1) + (x2 * w2) + b
    x1w1 = x1 * w1
    x2w2 = x2 * w2
    x1w1_x2w2 = x1w1 + x2w2
    value = x1w1_x2w2 + b
    # applying the activation function
    output = value.tanh()

    # give the labels to the intermediate nodes for better visualization
    x1w1.label = "x1 * w1"
    x2w2.label = "x2 * w2"
    x1w1_x2w2.label = "(x1 * w1) + (x2 * w2)"
    value.label = "(x1 * w1) + (x2 * w2) + b"
    output.label = "output"

    graph = build_computation_graph(output)
    plot_computation_graph(graph)


@app.command()
def neuron():
    n = Neuron(2)
    x = [2.0, 3.0]

    # forward the inputs x to a single Neuron object
    neuron_output = n(x)
    graph = build_computation_graph(neuron_output)
    plot_computation_graph(graph)


@app.command()
def layer():
    nn_layer = Layer(2, 5)  # 2 inputs and 5 outputs
    x = [2.0, 3.0]
    # forward the inputs to a single layer object
    layer_outputs = nn_layer(x)
    print(f"\t{Fore.CYAN}Layer outputs:{Fore.YELLOW} {[f'{v.data:.4f}' for v in layer_outputs]}")


@app.command()
def mlp(steps: int = 10, step_size: float = 0.01):
    features = [
        [2.0, 3.0, -1.0],  # sample 1
        [3.0, -1.0, 0.5],  # sample 2
        [0.5, 1.0, 1.0],  # sample 3
        [1.0, 1.0, -1.0],  # sample 4
    ]

    true_labels = [1.0, -1.0, -1.0, 1.0]  # targets
    net = MLP(3, [4, 4, 1])  # redefine the NN

    print(f"\n{Fore.YELLOW}>>> {Fore.GREEN}The network")
    print(f"\t{Fore.CYAN}Architecture:{Fore.YELLOW} {net}")
    print(f"\t{Fore.CYAN}Size: {Fore.YELLOW} {len(net.parameters())}")

    print(f"{Fore.CYAN}{"-" * 100}")
    print(f"\n{Fore.YELLOW}>>> {Fore.GREEN}Optimization steps:")

    # Optimization:
    for step in range(steps):
        # forward pass
        predictions = [net(x) for x in features]
        # sum of square error
        loss = sum((pred - truth) ** 2 for truth, pred in zip(true_labels, predictions, strict=False))

        # backward pass
        for p in net.parameters():  # zero the gradients first
            p.grad = 0.0
        loss.backward()

        # parameters updates
        for p in net.parameters():
            p.data += -step_size * p.grad

        print(f"\t[{step}] loss: {Fore.YELLOW}{loss.data:.4f}")

    print(f"{Fore.CYAN}{"-" * 100}")
    print(f"\n>>> {Fore.GREEN}Predictions:")

    print(f"\t{Fore.BLUE}{[f'{v:.4f}' for v in true_labels]}")
    print(f"\t{Fore.YELLOW}{[f'{pred.data:.4f}' for pred in predictions]}")
    print()


if __name__ == "__main__":
    app()
