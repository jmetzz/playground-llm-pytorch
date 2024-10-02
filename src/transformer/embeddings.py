import torch
from torch import Tensor, nn

from common.constants import END_TOKEN, PADDING_TOKEN, START_TOKEN
from common.utils import get_default_device


def tokenize(  # noqa: PLR0913
    sentence: str,
    language_index: dict[str, int],
    max_seq_length: int = 200,
    *,
    start_token: str = START_TOKEN,
    end_token: str = END_TOKEN,
    padding_token: str = PADDING_TOKEN,
) -> Tensor:
    sentence_index = [language_index[token] for token in sentence.split(" ")]
    if start_token:
        sentence_index.insert(0, language_index[start_token])
    if end_token:
        sentence_index.append(language_index[end_token])

    padding_size = max_seq_length - len(sentence_index)
    padding_arr = [language_index[padding_token]] * padding_size
    return Tensor(sentence_index + padding_arr)


class SentenceEmbedding(nn.Module):
    """Generate the embedding representing a given sentence"""

    def __init__(  # noqa: PLR0913
        self,
        model_dim: int,
        max_sequence_length: int,
        language_index: dict[str, int],
        dropout: float = 0.1,
        *,
        start_token: str = START_TOKEN,
        end_token: str = END_TOKEN,
        padding_token: str = PADDING_TOKEN,
    ):
        super().__init__()
        self.vocab_size = len(language_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, model_dim)
        self.language_index = language_index
        self.position_encoder = PositionalEncodingLayer(model_dim, max_sequence_length)
        self.dropout_layer = nn.Dropout(dropout)
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token

    def forward(self, tokens: Tensor):
        embeddings = self.batch_tokenize(tokens)
        embeddings = self.embedding(embeddings)
        position = self.position_encoder().to(get_default_device())
        return self.dropout_layer(embeddings + position)

    def batch_tokenize(self, batch: list[str]) -> Tensor:
        tokenized = [
            tokenize(
                batch[sentence_num],
                self.language_index,
                self.max_sequence_length,
                self.start_token,
                self.end_token,
                self.padding_token,
            )
            for sentence_num in range(len(batch))
        ]

        tokenized = torch.stack(tokenized)
        return tokenized.to(get_default_device())


class PositionalEncodingLayer(nn.Module):
    def __init__(self, model_dim: int, max_sequence_length: int):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.model_dim = model_dim

    def forward(self) -> Tensor:
        even_i = torch.arange(0, self.model_dim, 2).float()
        denominator = torch.pow(10000, even_i / self.model_dim)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_position_encoding = torch.sin(position / denominator)
        odd_position_encoding = torch.cos(position / denominator)
        stacked = torch.stack([even_position_encoding, odd_position_encoding], dim=2)
        return torch.flatten(stacked, start_dim=1, end_dim=2)
