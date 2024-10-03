import logging

import torch
import torch.nn.functional as torch_func
from torch import Tensor, nn

LOGGER = logging.getLogger()


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

    # Unscaled dot-product of queries and keys
    # resulting in a tensor of shape [batch, heads, seq_len, seq_len]
    attention_scores = queries @ keys.transpose(-2, -1)

    if mask is not None:
        # Apply the mask to avoid attending to certain positions (e.g., future positions in the decoder).
        # It also broadcast to every batch and every head.
        # Using in-place masking here for performance reasons
        attention_scores.masked_fill_(mask.bool(), -torch.inf)

    # Apply softmax to normalized scores.
    # sqrt(d_k) scaling factor --> this reduces variance and put the values in a zero mean and std 1
    # for better stabilization of the learning process
    scaling_factor = queries.size(-1) ** 0.5  # mathematically equivalent to math.sqrt(dim)

    attention_weights = torch_func.softmax(
        attention_scores / scaling_factor, dim=-1
    )  # [batch, heads, seq_len, seq_len]

    # Compute the context vector as a weighted sum of the values
    context_vectors = attention_weights @ values  # [batch, heads, seq_len, head_dim]
    LOGGER.debug(
        "Scaled dot product attention",
        extra={
            "attention_scores": attention_scores.shape,  # [batch_size, num_tokens, num_tokens]
            "attention_weights": attention_weights.shape,  # [batch_size, num_tokens, num_tokens]
            "context_vectors": context_vectors.shape,  # [batch_size, num_tokens, qkv_dim]
        },
    )
    return context_vectors, attention_weights


class MultiHeadSelfAttention(nn.Module):
    """
    The attention mechanism projects the input embeddings into query, key, and value vectors,
    applies scaled dot-product attention, and then recombines the results.
    """

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.model_dim = model_dim  # Typically the embedding dimension, e.g., 512
        self.num_heads = num_heads  # e.g., 8 heads
        self.head_dim = model_dim // num_heads  # e.g., 64 dimensions per head

        if model_dim % num_heads != 0:
            # Ensure model_dim is divisible by num_heads
            raise ValueError("model_dim must be divisible by num_heads.")

        # Linear layers to project input embeddings into queries, keys, and values
        # These layers could be combined into one to improve computational performance
        # as in `self.qkv_layer = nn.Linear(model_dim, 3 * model_dim)`.
        # However, this implementation focuses on readability instead of performance.
        self.query_layer = nn.Linear(model_dim, model_dim)
        self.key_layer = nn.Linear(model_dim, model_dim)
        self.value_layer = nn.Linear(model_dim, model_dim)

        # Output projection layer after attention
        self.output_layer = nn.Linear(model_dim, model_dim)

    def forward(self, embeddings: Tensor, mask: Tensor = None) -> Tensor:
        batch_size, seq_length, _ = embeddings.size()  # Input shape: [batch_size, seq_len, model_dim]

        # Project the input embeddings into queries, keys, and values if shape [batch_size, seq_len, model_dim]
        queries = self.query_layer(embeddings)
        keys = self.key_layer(embeddings)
        values = self.value_layer(embeddings)

        # Reshape the projected queries, keys, and values for multi-head attention,
        # which should result in shape [batch_size, num_heads, seq_len, head_dim]
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply scaled dot-product attention
        context_vector, attention_weights = scaled_dot_product_attention(queries, keys, values, mask)

        # Reshape context vector back to [batch_size, seq_length, model_dim]
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch_size, seq_length, self.model_dim)

        # Apply the final linear projection layer to the combined heads
        output = self.output_layer(context_vector)  # [batch_size, seq_len, model_dim]

        msg = {
            "model parameters": {
                f"{embeddings.size()=}",
                f"{queries.size()=}",
                f"{keys.size()=}",
                f"{values.size()=}",
                f"{attention_weights.size()=}",
                f"{context_vector.size()=}",
                f"{output.size()=}",
            }
        }
        logging.debug("MultiHeadAttention parameters", extra=msg)

        return output


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
