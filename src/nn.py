"""Main package module."""

import logging

import torch

LOGGER = logging.getLogger()


def build_embeddings(tokens: torch.Tensor, vocab_size: int, embeddings_dim: int, seq_length: int) -> torch.Tensor:
    token_embedding_layer = torch.nn.Embedding(vocab_size, embeddings_dim)
    token_embeddings = token_embedding_layer(tokens)

    positional_embedding_layer = torch.nn.Embedding(seq_length, embeddings_dim)
    # torch.arange(...) - placeholder vector which contains a sequence
    # of numbers 0, 1, ..., up to the maximum input `length - 1`
    positional_embeddings = positional_embedding_layer(torch.arange(seq_length))
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
    input_embeddings: torch.Tensor, embeddings_dim: int, qkv_dim: int, seed: int = 123
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)

    w_query = torch.nn.Parameter(torch.rand(embeddings_dim, qkv_dim), requires_grad=True)
    w_key = torch.nn.Parameter(torch.rand(embeddings_dim, qkv_dim), requires_grad=True)
    w_value = torch.nn.Parameter(torch.rand(embeddings_dim, qkv_dim), requires_grad=True)

    queries = input_embeddings @ w_query
    keys = input_embeddings @ w_key
    values = input_embeddings @ w_value

    LOGGER.debug(
        "QKV matrices built.",
        extra={"queries_shape": queries.shape, "keys_shape": keys.shape, "values_shape": values.shape},
    )

    return queries, keys, values


class SelfAttentionV1(torch.nn.Module):
    def __init__(self, embeddings_dim: int, output_dim: int, seed: int = 123) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self._w_query = torch.nn.Parameter(torch.rand(embeddings_dim, output_dim))
        self._w_key = torch.nn.Parameter(torch.rand(embeddings_dim, output_dim))
        self._w_value = torch.nn.Parameter(torch.rand(embeddings_dim, output_dim))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings shape: (batch_size, seq_length, embedding_dim)
        queries = embeddings @ self._w_query  # Shape: (batch_size, seq_length, output_dim)
        keys = embeddings @ self._w_key  # Shape: (batch_size, seq_length, output_dim)
        values = embeddings @ self._w_value  # Shape: (batch_size, seq_length, output_dim)

        # Compute attention scores
        # swap the last two dimensions of keys tensor.
        # transpose(-2, -1) operation will swap the second-to-last dimension with the last dimension.
        attention_scores = queries @ keys.transpose(-2, -1)  # Shape: (batch_size, output_dim, seq_length)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        return attention_weights @ values  # context vectors


class SelfAttentionV2(torch.nn.Module):
    def __init__(
        self, embeddings_dim: int, output_dim: int, *, qkv_bias: bool = False, seed: int | None = None
    ) -> None:
        super().__init__()
        if seed:
            torch.manual_seed(seed)
        self._w_query = torch.nn.Linear(embeddings_dim, output_dim, bias=qkv_bias)
        self._w_key = torch.nn.Linear(embeddings_dim, output_dim, bias=qkv_bias)
        self._w_value = torch.nn.Linear(embeddings_dim, output_dim, bias=qkv_bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings shape: (batch_size, seq_length, embedding_dim)
        queries = self._w_query(embeddings)
        keys = self._w_key(embeddings)
        values = self._w_value(embeddings)

        # Compute attention scores
        # swap the last two dimensions of keys tensor.
        # transpose(-2, -1) operation will swap the second-to-last dimension with the last dimension.
        attention_scores = queries @ keys.transpose(-2, -1)  # Shape: (batch_size, output_dim, seq_length)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        return attention_weights @ values  # context vectors


class CausalAttentionV1(torch.nn.Module):
    def __init__(
        self, embedding_dim: int, output_dim: int, dropout: float, *, qkv_bias: bool = False, seed: int = 123
    ) -> None:
        super().__init__()
        if seed:
            torch.manual_seed(seed)

        self._output_dim = output_dim

        self._w_query = torch.nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self._w_key = torch.nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self._w_value = torch.nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self._dropout = torch.nn.Dropout(dropout)

        # #TODO: For better performance, the mask could be buffered here.
        #   Something like `self.register_buffer("mask", torch.triu(torch.ones(output_dim, output_dim), diagonal=1))`
        #   Buffers are automatically moved to the appropriate device (CPU or GPU) in runtime.

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        keys = self._w_key(embeddings)
        queries = self._w_query(embeddings)
        values = self._w_value(embeddings)

        attention_scores = torch.bmm(queries, keys.transpose(-2, -1))

        # Apply mask to prevent attending to future positions
        _, num_tokens, _ = embeddings.shape
        mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1).bool()

        # In-place masking of future positions in the attention scores
        attention_scores.masked_fill_(mask.bool(), -torch.inf)

        # Apply softmax to normalized scores
        scaling_factor = keys.shape[-1] ** 0.5  # scale by sqrt(qkv_dim)
        attention_weights = torch.softmax(attention_scores / scaling_factor, dim=-1)

        # Apply dropout to attention weights
        attention_weights = self._dropout(attention_weights)

        # Compute the context vectors (weighted sum of values)
        context_vectors = attention_weights @ values

        LOGGER.debug(
            "Forward pass.",
            extra={
                "attention_scores": attention_scores.shape,  # [batch_size, num_tokens, num_tokens]
                "attention_weights": attention_weights.shape,  # [batch_size, num_tokens, num_tokens]
                "context_vectors": context_vectors.shape,  # [batch_size, num_tokens, qkv_dim]
            },
        )
        return context_vectors


class MultiHeadAttentionV1(torch.nn.Module):
    """Multiple heads of causal self-attention in parallel."""

    def __init__(
        self,
        embeddings_dim: int,
        head_dim: int,
        num_heads: int,
        dropout: float = 0.2,
        *,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        heads = [
            CausalAttentionV1(embedding_dim=embeddings_dim, output_dim=head_dim, dropout=dropout, qkv_bias=qkv_bias)
            for _ in range(num_heads)
        ]
        self.heads = torch.nn.ModuleList(heads)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # concatenates over the channel dimension
        res = [h(embeddings) for h in self.heads]
        return torch.cat(res, dim=-1)
