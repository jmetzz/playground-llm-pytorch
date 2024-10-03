from torch import Tensor, nn

from transformer.attention import MultiHeadCrossAttention, MultiHeadSelfAttention
from transformer.normalization import NormalizationLayer


class FeedForwardBlock(nn.Module):
    """Feed-forward network used in the Transformer model, consisting of two linear layers with
    a ReLU activation and dropout in between.

    Args:
        model_dim (int): Dimensionality of the input and output embeddings.
        hidden_dim (int): Dimensionality of the hidden layer in the feed-forward network.
        dropout (float): Dropout probability applied after the ReLU activation.
    """

    def __init__(self, model_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, hidden_dim)  # First linear layer: [model_dim, num_hidden]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, model_dim)  # Second linear layer: [num_hidden, model_dim]

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through the feed-forward network.

        Args:
            inputs (Tensor): Input tensor to the feed-forward network
                    (shape: [batch_size, seq_length, model_dim]).

        Returns:
            Tensor: Output tensor after processing through the feed-forward network
                    (shape: [batch_size, seq_length, model_dim]).
        """
        embeddings = self.linear1(inputs)  # Linear transformation: [batch_size, seq_length, num_hidden]
        embeddings = self.relu(embeddings)  # ReLU activation: [batch_size, seq_length, num_hidden]
        embeddings = self.dropout(embeddings)  # Apply dropout: [batch_size, seq_length, num_hidden]
        return self.linear2(embeddings)  # Final linear transformation: [batch_size, seq_length, model_dim]


class EncoderLayer(nn.Module):
    """A single layer of the Transformer encoder.

    Consisting of self-attention and a feed-forward network
    with normalization and dropout applied at each step.

    Args:
        model_dim (int): The dimensionality of the input and output embeddings.
        hidden_size (int): The dimensionality of the hidden layer in the feed-forward network.
        num_heads (int): The number of attention heads in the multi-head attention mechanism.
        dropout (float): Dropout probability applied after each sub-layer.
    """

    def __init__(self, model_dim: int, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadSelfAttention(model_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = NormalizationLayer(model_dim)

        self.ffn = FeedForwardBlock(model_dim, hidden_size, dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm2 = NormalizationLayer(model_dim)

    def forward(self, encoder_inputs: Tensor, self_attention_mask: Tensor) -> Tensor:
        """Forward pass through the encoder layer.

        Args:
            encoder_inputs (Tensor): Input tensor to the encoder layer
                    (shape: [batch_size, seq_length, model_dim]).
            self_attention_mask (Tensor): Mask to prevent attending to future tokens
                    or padding (shape: [batch_size, seq_length, seq_length]).

        Returns:
            Tensor: Output tensor after processing through self-attention and the feed-forward network.
        """
        # Self-attention mechanism
        residuals = encoder_inputs.clone()  # Save for residual connection
        embeddings = self.attention(encoder_inputs, self_attention_mask)
        embeddings = self.dropout1(embeddings)
        embeddings = self.norm1(embeddings + residuals)

        # Feed-forward network
        residuals = embeddings.clone()  # Save for residual connection
        embeddings = self.ffn(embeddings)
        embeddings = self.dropout2(embeddings)
        return self.norm2(embeddings + residuals)


class DecoderLayer(nn.Module):
    """A single layer in the Transformer decoder, consisting of self-attention, encoder-decoder attention,
    and a feed-forward network with normalization and dropout applied at each step.

    Args:
        model_dim (int): The dimensionality of the input and output embeddings.
        hidden_size (int): The dimensionality of the hidden layer in the feed-forward network.
        num_heads (int): The number of attention heads in the multi-head attention mechanism.
        dropout (float): Dropout probability applied after each sub-layer.
    """

    def __init__(self, model_dim: int, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(model_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = NormalizationLayer(parameters_shape=[model_dim])

        self.encoder_decoder_attention = MultiHeadCrossAttention(model_dim, num_heads)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = NormalizationLayer(parameters_shape=[model_dim])

        self.ffn = FeedForwardBlock(model_dim, hidden_size, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = NormalizationLayer(parameters_shape=[model_dim])

    def forward(self, decoder_inputs: Tensor, encoder_outputs: Tensor, self_attention_mask: Tensor) -> Tensor:
        """Forward pass through the decoder layer.

        Args:
            decoder_inputs (Tensor): Input tensor to the decoder layer
                    (shape: [batch_size, seq_length, model_dim]).
            encoder_outputs (Tensor): Output tensor from the encoder to attend over
                    (shape: [batch_size, seq_length, model_dim]).
            self_attention_mask (Tensor): Mask to prevent attending to future tokens
                    during self-attention (shape: [batch_size, seq_length, seq_length]).

        Returns:
            Tensor: Output tensor after processing through self-attention,
                    encoder-decoder attention, and the feed-forward network.
        """
        # Self-attention mechanism
        residual = decoder_inputs.clone()  # Save for residual connection
        y = self.self_attention(decoder_inputs, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.norm1(y + residual)

        # Cross-attention mechanism (encoder-decoder attention)
        residual = y.clone()  # Save for residual connection
        y = self.encoder_decoder_attention(encoder_outputs, y, mask=None)
        y = self.dropout2(y)
        y = self.norm2(y + residual)

        # Feed-forward network
        residual = y.clone()  # Save for residual connection
        y = self.ffn(y)
        y = self.dropout3(y)
        return self.norm3(y + residual)
