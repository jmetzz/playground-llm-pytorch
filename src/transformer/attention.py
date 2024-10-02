import logging

import torch
import torch.nn.functional as torch_func
from torch import Tensor, nn


def scaled_dot_product_attention(
    queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor = None
) -> tuple[Tensor, Tensor]:
    """
    Computes the scaled dot-product attention.

    This function calculates attention scores between query and key vectors,
    applies an optional mask, and returns weighted values and the attention matrix.

    Args:
        queries (Tensor): The query matrix of shape (batch_size, num_heads, seq_length, qkv_dim), representing the
            vectors that specify what the model is looking for.
        keys (Tensor): The key matrix of shape (batch_size, num_heads, seq_length, qkv_dim), representing the vectors
            that describe what each element has to offer in the attention mechanism.
        values (Tensor): The value matrix of shape (batch_size, num_heads, seq_length, qkv_dim), representing the
            actual content to attend to based on the attention weights.
        mask (Tensor, optional): A tensor used to mask certain positions (e.g., future positions in causal attention)
            during the attention score computation. Broadcasts over the batch and heads. Defaults to None.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing:
            - The output values of shape (batch_size, num_heads, seq_length, qkv_dim), representing the result
              of applying attention weights to the values.
            - The attention matrix of shape (batch_size, num_heads, seq_length, seq_length), representing
              the attention scores after applying softmax.
    """
    # sqrt(d_k) scaling factor --> this reduces variance and put the values in a zero mean and std 1
    # for better stabilization of the learning process

    # example dimensions: q, k, v = 30x8x200x64
    scaling_factor = queries.size()[-1] ** 0.5  # mathematically equivalent to math.sqrt(dim)
    scaled = torch.matmul(queries, keys.transpose(-1, -2)) / scaling_factor  # 30 x 8 x 200 x 200
    if mask:
        # Apply the mask to avoid attending to certain positions (e.g., future positions in the decoder)
        # it also broadcast to every batch and every head.
        # resulting in 30 x 8 x 200 x 200 masked
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)

    # Calculate attention probabilities via softmax
    attention = torch_func.softmax(scaled, dim=-1)  # 30 x 8 x 200 x 200

    # Compute the context vector as a weighted sum of the values
    context_vector = torch.matmul(attention, values)  # 30 x 8 x 200 x 64
    return context_vector, attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.model_dim = model_dim  # 512
        self.num_heads = num_heads  # 8
        self.head_dim = model_dim // num_heads  # 64
        self.qkv_layer = nn.Linear(model_dim, 3 * model_dim)  # 512 x 1536
        self.linear_layer = nn.Linear(model_dim, model_dim)  # 512 x 512

    def forward(self, embeddings: Tensor, mask: Tensor = None) -> Tensor:
        batch_size, sequence_length, _ = embeddings.size()  # input: 30 x 200 x 512
        qkv = self.qkv_layer(embeddings)  # 30 x 200 x 1536
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)  # 30 x 200 x 8 x 192
        qkv = qkv.permute(0, 2, 1, 3)  # 30, 8, 200, 192
        # now break the tensor in 3 according to the last dimension. Each is 30 x 8 x 200 x 64
        queries, keys, values = qkv.chunk(3, dim=-1)
        context_vector, attention = scaled_dot_product_attention(queries, keys, values, mask)
        # attention: 30 x 8 x 200 x 200
        # values: 30 x 8 x 200 x 64
        context_vector = context_vector.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )  # 30 x 200 x 512
        result = self.linear_layer(context_vector)  # output: 30 x 200 x 512

        msg = {
            "embeddings.size()": embeddings.size(),
            "qkv_size()": qkv.size(),
            "queries size": queries.size(),
            "keys size": keys.size(),
            "values size": values.size(),
            "attention.size()": attention.size(),
            "context_vector.size()": context_vector.size(),
            "result.size()": result.size(),
        }
        logging.debug("Parameters", extra=msg)

        return result


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.model_dim = model_dim  # 512
        self.num_heads = num_heads  # 8
        self.heads_dim = model_dim // num_heads  # 64
        self.kv_layer = nn.Linear(model_dim, 2 * model_dim)  # 512 x 1024
        self.q_layer = nn.Linear(model_dim, model_dim)  # 512 x 512
        self.linear_layer = nn.Linear(model_dim, model_dim)  # 512 x 512

    def forward(self, embeddings: Tensor, mask: Tensor = None) -> Tensor:
        batch_size, sequence_length, _ = embeddings.size()  # input: 30 x 200 x 512
        keys_values = self.kv_layer(embeddings)  # 30 x 200 x 1024
        queries = self.q_layer(embeddings)  # 30 x 200 x 512
        keys_values = keys_values.reshape(
            batch_size, sequence_length, self.num_heads, 2 * self.heads_dim
        )  # 30 x 200 x 8 x 128
        keys_values = keys_values.permute(0, 2, 1, 3)  # 30 x 8 x 200 x 128
        queries = queries.reshape(batch_size, sequence_length, self.num_heads, self.heads_dim)  # 30 x 200 x 8 x 64
        queries = queries.permute(0, 2, 1, 3)  # 30 x 8 x 200 x 64

        # now break the tensor in 3 according to the last dimension. Each is 30 x 8 x 200 x 64
        keys, values = keys_values.chunk(2, dim=-1)  # k: 30 x 8 x 200 x 64  | v :30 x 8 x 200 x 64
        # We don't need the mask for cross attention. It should be None.
        context_vectors, attention = scaled_dot_product_attention(queries, keys, values, mask)
        # attention: 30 x 8 x 200 x 200
        # values: 30 x 8 x 200 x 64

        context_vectors = context_vectors.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.num_heads * self.heads_dim
        )  # 30 x 200 x 512

        result = self.linear_layer(context_vectors)  # output: 30 x 200 x 512
        # input and output shape matches :)

        msg = {
            "embeddings.size()": embeddings.size(),
            "keys_values size()": keys_values.size(),
            "queries size": queries.size(),
            "keys size": keys.size(),
            "values size": values.size(),
            "attention.size()": attention.size(),
            "context_vectors.size()": context_vectors.size(),
            "result.size()": result.size(),
        }
        logging.debug("Parameters", extra=msg)

        return result
