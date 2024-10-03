"""Command-line interface."""

import logging

import colorama
import torch
import typer
from torch import Tensor

from common.constants import END_TOKEN, PADDING_TOKEN, START_TOKEN
from transformer import tokenizers
from transformer.attention import MultiHeadCrossAttention, MultiHeadSelfAttention
from transformer.embeddings import SentenceEmbedding
from transformer.modules import EncoderLayer, FeedForwardBlock, NormalizationLayer

colorama.init(autoreset=True)

app = typer.Typer()

LOGGER = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

# Sample vocabulary for testing purposes
LANGUAGE_INDEX = {PADDING_TOKEN: 0, START_TOKEN: 1, END_TOKEN: 2, "hello": 3, "world": 4}
SENTENCES_SAMPLE = [
    "hello world",
    f"{START_TOKEN} world hello",
    f"world hello {END_TOKEN}",
    f"{START_TOKEN} hello world {END_TOKEN} {PADDING_TOKEN}",
    f"world {PADDING_TOKEN} hello ",
    f"{PADDING_TOKEN} world hello",
    f"{START_TOKEN} hello world world world world world {END_TOKEN}",
    f"{START_TOKEN} hello world {END_TOKEN} {PADDING_TOKEN} {PADDING_TOKEN} {PADDING_TOKEN} {PADDING_TOKEN}",
]


def generate_dummy_inputs(
    batch_size: int = 10, seq_length: int = 5, model_dim: int = 32, num_heads: int = 4, qkv_dim: int | None = None
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
    # Generate random encoder inputs
    encoder_inputs = torch.rand(batch_size, seq_length, model_dim)

    # Generate self-attention mask
    self_attention_mask = torch.triu(torch.zeros(batch_size, num_heads, seq_length, seq_length), diagonal=1).bool()

    if qkv_dim:
        return encoder_inputs, self_attention_mask, build_dummy_qkv_matrices(encoder_inputs, qkv_dim=qkv_dim)

    return encoder_inputs, self_attention_mask


def build_dummy_qkv_matrices(input_embeddings: Tensor, qkv_dim: int = 2) -> tuple[Tensor, Tensor, Tensor]:
    model_dim = input_embeddings.size(-1)
    w_query = torch.nn.Parameter(torch.rand(model_dim, qkv_dim), requires_grad=False)
    w_key = torch.nn.Parameter(torch.rand(model_dim, qkv_dim), requires_grad=False)
    w_value = torch.nn.Parameter(torch.rand(model_dim, qkv_dim), requires_grad=False)

    queries = input_embeddings @ w_query
    keys = input_embeddings @ w_key
    values = input_embeddings @ w_value

    return queries, keys, values


@app.command()
def tokenize(seq_length: int = 5):
    for sentence in SENTENCES_SAMPLE:
        print(
            tokenizers.tokenize(
                sentence=sentence,
                language_index=LANGUAGE_INDEX,
                max_seq_length=seq_length,
                start_token=START_TOKEN,
                end_token=END_TOKEN,
                padding_token=PADDING_TOKEN,
            )
        )


@app.command()
def tokenize_batch(seq_length: int = 5):
    # Tokenize the sentence manually (convert words to indices)
    result = tokenizers.tokenize_batch(
        batch=SENTENCES_SAMPLE,
        language_index=LANGUAGE_INDEX,
        max_seq_length=seq_length,
        start_token=START_TOKEN,
        end_token=END_TOKEN,
        padding_token=PADDING_TOKEN,
    )

    print(result)


@app.command()
def embed(model_dim: int = 32, seq_length: int = 5, dropout: float = 0.2):
    # Sample vocabulary for testing purposes
    # language_index = {PADDING_TOKEN: 0, START_TOKEN: 1, END_TOKEN: 2, "hello": 3, "world": 4}
    # sentences = ["hello world", f"{START_TOKEN} hello world {END_TOKEN} {PADDING_TOKEN}"]

    # Tokenize the sentence manually (convert words to indices)
    tokenized_batch = tokenizers.tokenize_batch(
        batch=SENTENCES_SAMPLE,
        language_index=LANGUAGE_INDEX,
        max_seq_length=seq_length,
        start_token=START_TOKEN,
        end_token=END_TOKEN,
        padding_token=PADDING_TOKEN,
    )

    # The SentenceEmbedding class expects a list of tokenized sentences (not raw strings)
    sentence_embeddings = SentenceEmbedding(
        model_dim=model_dim,
        max_sequence_length=seq_length,
        language_index=LANGUAGE_INDEX,
        dropout=dropout,
        start_token=START_TOKEN,
        end_token=END_TOKEN,
        padding_token=PADDING_TOKEN,
    )
    # Run the model
    embeddings = sentence_embeddings(tokenized_batch)  # Pass the batch of sentences
    print(f"{embeddings.shape=}")
    print(embeddings[0])


@app.command()
def attention_multihead(
    batch_size: int = 10, seq_length: int = 5, model_dim: int = 32, num_heads: int = 4, *, use_mask: bool = False
):
    token_embeddings, self_attention_mask = generate_dummy_inputs(batch_size, seq_length, model_dim)

    model = MultiHeadSelfAttention(model_dim=model_dim, num_heads=num_heads)

    result = model.forward(token_embeddings, self_attention_mask) if use_mask else model.forward(token_embeddings)
    print(result)


@app.command()
def attention_multihead_cross(batch_size: int = 10, seq_length: int = 5, embeddings_dim: int = 32):
    token_embeddings = torch.randn((batch_size, seq_length, embeddings_dim))

    model = MultiHeadCrossAttention(model_dim=512, num_heads=8)
    out = model.forward(token_embeddings)
    print(out)


@app.command()
def norm_layer(batch_size: int = 10, seq_length: int = 5, embeddings_dim: int = 32):
    embedding = torch.randn(batch_size, seq_length, embeddings_dim)
    layer_norm = NormalizationLayer(embeddings_dim)
    print(layer_norm(embedding))


@app.command()
def ff_block(
    batch_size: int = 10,
    seq_length: int = 5,
    num_heads: int = 4,
    input_dim: int = 32,
    hidden_size: int = 4,
    dropout: float = 0.2,
):
    token_embeddings, _ = generate_dummy_inputs(
        batch_size=batch_size, seq_length=seq_length, model_dim=input_dim, num_heads=num_heads
    )
    ff_layer_block = FeedForwardBlock(input_dim, hidden_size, dropout)

    print(ff_layer_block(token_embeddings))



if __name__ == "__main__":
    app()
