"""Command-line interface."""

import logging

import colorama
import torch
import typer

from transformer.attention import MultiHeadCrossAttention, MultiHeadSelfAttention
from transformer.normalization import NormalizationLayer

colorama.init(autoreset=True)

app = typer.Typer()

LOGGER = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


@app.command()
def attention_multihead(batch_size: int = 30, seq_length: int = 5, embeddings_dim: int = 32):
    token_embeddings = torch.randn((batch_size, seq_length, embeddings_dim))

    model = MultiHeadSelfAttention(model_dim=512, num_heads=8)
    out = model.forward(token_embeddings)
    print(out)


@app.command()
def attention_multihead_cross(batch_size: int = 30, seq_length: int = 5, embeddings_dim: int = 32):
    token_embeddings = torch.randn((batch_size, seq_length, embeddings_dim))

    model = MultiHeadCrossAttention(model_dim=512, num_heads=8)
    out = model.forward(token_embeddings)
    print(out)


@app.command()
def norm_layer(batch_size: int = 30, seq_length: int = 5, embeddings_dim: int = 32):
    embedding = torch.randn(batch_size, seq_length, embeddings_dim)
    layer_norm = NormalizationLayer(embeddings_dim)
    print(layer_norm(embedding))


if __name__ == "__main__":
    app()
