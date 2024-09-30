"""Main package module."""

import logging

import torch

LOGGER = logging.getLogger()


def build_embeddings(inputs: torch.Tensor, vocab_size: int, output_dim: int, context_length: int) -> torch.Tensor:
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)

    positional_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    # torch.arange(...) - placeholder vector which contains a sequence
    # of numbers 0, 1, ..., up to the maximum input `length - 1`
    positional_embeddings = positional_embedding_layer(torch.arange(context_length))
    input_embeddings = token_embeddings + positional_embeddings

    LOGGER.debug(
        "Embeddings tensors built.",
        extra={
            "token_embeddings": token_embeddings.shape,
            "positional_embeddings": positional_embeddings.shape,
            "input_embeddings": input_embeddings.shape,
        },
    )
    return input_embeddings
