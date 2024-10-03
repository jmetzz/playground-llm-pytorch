from torch import Tensor, nn

from common.constants import END_TOKEN, PADDING_TOKEN, START_TOKEN
from transformer.embeddings import SentenceEmbedding
from transformer.modules import EncoderLayer


class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs) -> Tensor:
        embeddings, self_attention_mask = inputs
        for module in self._modules.values():
            embeddings = module(embeddings, self_attention_mask)
        return embeddings


class TransformerEncoder(nn.Module):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        model_dim: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_sequence_length: int,
        language_index: dict[str, int],
        *,
        start_token: str = START_TOKEN,
        end_token: str = END_TOKEN,
        padding_token: str = PADDING_TOKEN,
    ):
        super().__init__()
        self.sentence_embeddings = SentenceEmbedding(
            model_dim, max_sequence_length, language_index, dropout, start_token, end_token, padding_token
        )
        self.layers = SequentialEncoder(
            *[EncoderLayer(model_dim, hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, tokens: Tensor, self_attention_mask: Tensor) -> Tensor:
        embeddings = self.sentence_embeddings(tokens)
        return self.layers(embeddings, self_attention_mask)
