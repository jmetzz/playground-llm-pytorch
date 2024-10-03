import torch
from torch import Tensor, nn

from transformer.attention import MultiHeadCrossAttention, MultiHeadSelfAttention


class NormalizationLayer(nn.Module):
    def __init__(self, model_dim: int, eps: int = 1e-5):
        """
        Normalizes the input embeddings across the feature dimension.

        Args:
            model_dim (int): The dimensionality of the feature (embedding) space,
                            which is the number of features per token.
            eps (float): A small value to prevent division by zero during normalization.
        """
        super().__init__()
        self.eps = eps  # to avoid division by zero
        self.gamma = nn.Parameter(torch.ones(model_dim))  # Learnable scale parameter
        self.beta = nn.Parameter(torch.zeros(model_dim))  # Learnable shift parameter

    def forward(self, embeddings: Tensor) -> Tensor:  # 30 x 200 x 512
        """
        Forward pass to normalize the input embeddings across the feature dimension.

        Args:
            embeddings (Tensor): Input tensor of shape [batch_size, seq_length, embedding_dim].

        Returns:
            Tensor: The normalized embeddings of the same shape.
        """
        mean = embeddings.mean(dim=-1, keepdim=True)  # Mean along model_dim
        variance = ((embeddings - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (variance + self.eps).sqrt()  # use the eps for numerical stability
        normalized_embeddings = (embeddings - mean) / std

        # Apply learnable scale (gamma) and shift (beta)
        return self.gamma * normalized_embeddings + self.beta


class FeedForwardBlock(nn.Module):
    """Feed-forward network used in the Transformer model, consisting of two linear layers with
    a ReLU activation and dropout in between.

    Args:
        input_dim (int): Dimensionality of the input and output embeddings.
        output_dim (int): Dimensionality of the hidden layer in the feed-forward network.
        dropout (float): Dropout probability applied after the ReLU activation.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()  # Introduce non-linearity.
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(output_dim, input_dim)  # projects back to the original model_dim

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through the feed-forward network block.

        Args:
            inputs (Tensor): Input tensor of shape [batch_size, seq_length, embeddings_dim]).

        Returns:
            Tensor: Output tensor after processing through the feed-forward network
                    (shape: [batch_size, seq_length, embeddings_dim]).
        """
        embeddings = self.input_layer(inputs)
        embeddings = self.relu(embeddings)
        embeddings = self.dropout(embeddings)
        return self.output_layer(embeddings)


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
