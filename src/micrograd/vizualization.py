import io

import matplotlib.pyplot as plt
from graphviz import Digraph

from micrograd.engine import Operand

RANK_DIR_MAP = {"left-to-right": "LR", "top-to-bottom": "TB"}


def build_computation_graph(root: Operand, fmt: str = "svg", rankdir: str = "left-to-right") -> Digraph:
    if rankdir not in RANK_DIR_MAP:
        raise ValueError(f"rankdir must be one of '{RANK_DIR_MAP.keys()}'")

    nodes, edges = root.trace()
    graph = Digraph(format=fmt, graph_attr={"rankdir": RANK_DIR_MAP[rankdir]})

    for node in nodes:
        label = f"{{ {node.label} | data {node.data:.4f} | grad {node.grad:.4f} }}"
        graph.node(name=str(id(node)), label=label, shape="record")
        if node.src_operation:
            op_node_name = f"{id(node)}{node.src_operation}"
            graph.node(name=op_node_name, label=node.src_operation)
            graph.edge(op_node_name, str(id(node)))

    for n1, n2 in edges:
        graph.edge(str(id(n1)), str(id(n2)) + n2.src_operation)

    return graph


def export_graph(graph: Digraph, filename: str = "computation_graph") -> None:
    """Renders and saves the computation graph to a file."""
    graph.render(filename, cleanup=True)


def plot_computation_graph(graph: Digraph) -> None:
    """Plots the computation graph using Matplotlib."""

    img_bytes = graph.pipe(format="png")  # In-Memory Image

    # Create a figure and axis in Matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(plt.imread(io.BytesIO(img_bytes)), aspect="equal")
    plt.axis("off")
    plt.show()  # Display the plot
