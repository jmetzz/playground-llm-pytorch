import torch
from torch import Tensor, nn


class NormalizationLayer(nn.Module):
    def __init__(self, embeddings_dim: list, eps: int = 1e-5):
        """
        Normalizes the input embeddings across the feature dimension (embedding_dim).

        Args:
            embedding_dim (int): The size of the last dimension (feature dimension).
            eps (float): A small value to prevent division by zero during normalization.
        """
        super().__init__()
        self.eps = eps  # to avoid division by zero
        self.gamma = nn.Parameter(torch.ones(embeddings_dim))  # Learnable scale parameter
        self.beta = nn.Parameter(torch.zeros(embeddings_dim))  # Learnable shift parameter

    def forward(self, embeddings: Tensor) -> Tensor:  # 30 x 200 x 512
        """
        Forward pass to normalize the input embeddings across the feature dimension.

        Args:
            embeddings (Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).

        Returns:
            Tensor: The normalized embeddings of the same shape.
        """
        # Normalize only across the last dimension (embedding_dim)
        mean = embeddings.mean(dim=-1, keepdim=True)  # Mean along embedding_dim
        variance = ((embeddings - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (variance + self.eps).sqrt()  # use the eps for numerical stability
        normalized_embeddings = (embeddings - mean) / std

        # Apply learnable scale (gamma) and shift (beta)
        return self.gamma * normalized_embeddings + self.beta
