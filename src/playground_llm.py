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


def build_qkv_matrices(
    input_embeddings: torch.Tensor, input_dim: int, output_dim: int, seed: int = 123
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)

    w_query = torch.nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=True)
    w_key = torch.nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=True)
    w_value = torch.nn.Parameter(torch.rand(input_dim, output_dim), requires_grad=True)

    queries = input_embeddings @ w_query
    keys = input_embeddings @ w_key
    values = input_embeddings @ w_value

    LOGGER.debug(
        "QKV matrices built.",
        extra={"queries_shape": queries.shape, "keys_shape": keys.shape, "values_shape": values.shape},
    )

    return queries, keys, values
