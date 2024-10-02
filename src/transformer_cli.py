"""Command-line interface."""

import logging

import colorama
import torch
import typer

from transformer.attention import MultiHeadCrossAttention, MultiHeadSelfAttention

colorama.init(autoreset=True)

app = typer.Typer()

LOGGER = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


@app.command()
def attention_multihead(batch_size: int = 30, sequence_len: int = 5, dimension: int = 512):
    token_embeddings = torch.randn((batch_size, sequence_len, dimension))

    model = MultiHeadSelfAttention(model_dim=512, num_heads=8)
    out = model.forward(token_embeddings)
    print(out)


@app.command()
def attention_multihead_cross(batch_size: int = 30, sequence_len: int = 5, dimension: int = 512):
    token_embeddings = torch.randn((batch_size, sequence_len, dimension))

    model = MultiHeadCrossAttention(model_dim=512, num_heads=8)
    out = model.forward(token_embeddings)
    print(out)


if __name__ == "__main__":
    app()
