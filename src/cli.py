"""Command-line interface."""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import colorama
import tiktoken
import torch
import typer
from colorama import Fore, Style

from nn import SelfAttentionV1, SelfAttentionV2, build_embeddings, build_qkv_matrices
from splitters import punctuation_splitter, space_and_punctuation_splitter, space_splitter
from tokenizers import SimpleRegexTokenizerV1, SimpleRegexTokenizerV2
from utils.data import create_dataloader_v1, get_encoder_and_batch_iterator, load_text_file
from utils.io import print_attention_matrix

colorama.init(autoreset=True)

app = typer.Typer()

LOGGER = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


class SplitterOptions(StrEnum):
    SPACE = "space"
    PUNCTUATION = "punctuation"
    SPACE_AND_PUNCTUATION = "both"


@app.command()
def split_text(
    splitter: Annotated[
        SplitterOptions, typer.Option(case_sensitive=False, help="Choose how to split the text")
    ] = SplitterOptions.SPACE,
    length: int = 100,
    *,
    encode_spaces: bool = False,
):
    the_verdict_file = Path(__file__).parent.parent.joinpath("resources/the-verdict.txt")
    content = load_text_file(the_verdict_file)
    sentence = content[:length]
    if splitter == SplitterOptions.SPACE:
        tokens = space_splitter(sentence, encode_spaces=encode_spaces)
    elif splitter == SplitterOptions.PUNCTUATION:
        tokens = punctuation_splitter(sentence, encode_spaces=encode_spaces)
    elif splitter == SplitterOptions.SPACE_AND_PUNCTUATION:
        tokens = space_and_punctuation_splitter(sentence, encode_spaces=encode_spaces)
    print(f"Sentence: {sentence}")
    print(f"\nTokens: {tokens}")


@app.command()
def tokenize_v1():
    the_verdict_file = Path(__file__).parent.parent.joinpath("resources/the-verdict.txt")
    content = load_text_file(the_verdict_file)
    words = space_and_punctuation_splitter(content, encode_spaces=False)
    print(f"number of words: {len(words)}")

    # vocabulary = {token: idx for idx, token in enumerate(words)}
    print(f"vocabulary size: {len(set(words))}")
    tokenizer = SimpleRegexTokenizerV1(words)

    text = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride'
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))


@app.command()
def tokenize_v2():
    the_verdict_file = Path(__file__).parent.parent.joinpath("resources/the-verdict.txt")
    content = load_text_file(the_verdict_file)
    words = space_and_punctuation_splitter(content, encode_spaces=False)
    print(f"vocabulary size: {len(set(words))}")

    tokenizer = SimpleRegexTokenizerV2(words)
    text = SimpleRegexTokenizerV2.BLOCK_DELIMITER.join(
        ("Hello, do you like tea?", "In the sunlit terraces of the palace.")
    )
    print(text)
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))


@app.command()
def dataloader():
    _, data_loader = get_encoder_and_batch_iterator(
        Path(__file__).parent.parent.joinpath("resources/the-verdict.txt"),
        batch_size=8,
        stride=4,
        seq_length=4,
    )
    data_iter = iter(data_loader)
    batch_tokens, batch_labels = next(data_iter)

    print(Fore.CYAN + "Token IDs:")
    print(f"Batch inputs shape: {batch_tokens.shape}")  # [8, 1]
    print(f"{batch_tokens}")

    print(Fore.YELLOW + "Label IDs:")
    print(f"{batch_labels}")
    print()


@app.command()
def codec():
    encoder, data_loader = get_encoder_and_batch_iterator(
        Path(__file__).parent.parent.joinpath("resources/the-verdict.txt"),
        batch_size=1,
        stride=1,
        seq_length=4,
    )
    data_iter = iter(data_loader)
    print(Fore.CYAN + ">>> Testing data loader with parameters:")
    print(Fore.CYAN + "\t batch_size: 1")
    print(Fore.CYAN + "\t stride: 1")
    print(Fore.CYAN + "\t seq_length: 4")

    for idx in range(2):
        batch_tokens, batch_labels = next(data_iter)
        batch_tokens = batch_tokens.squeeze()
        batch_labels = batch_labels.squeeze()

        print(Fore.YELLOW + f">>> Batch #{idx}")
        print(Fore.LIGHTCYAN_EX + "encoded tokens: " + Fore.RESET + f"{batch_tokens}")
        print(Fore.LIGHTCYAN_EX + "encoded label: " + Fore.RESET + f"{batch_labels}")

        output = [encoder.decode([element]) for element in batch_tokens]
        print(Fore.BLUE + "decoded tokens: " + Fore.RESET + f"{output}")
        output = [encoder.decode([element]) for element in batch_labels]
        print(Fore.BLUE + "decoded label: " + Fore.RESET + f"{output}")


@app.command()
def embed(
    batch_size: int = 8,
    stride: int = 4,
    seq_length: int = 4,
    output_dim: int = 3,
):
    encoder, data_loader = get_encoder_and_batch_iterator(
        Path(__file__).parent.parent.joinpath("resources/the-verdict.txt"),
        batch_size=batch_size,
        stride=stride,
        seq_length=seq_length,
    )
    data_iter = iter(data_loader)

    batch_tokens, _ = next(data_iter)
    print(Fore.CYAN + "Batch tokens shape:")
    print(f"{batch_tokens.shape}")  # [batch_size, seq_length]
    print(Fore.CYAN + "Token IDs:")
    print(f"{batch_tokens}")

    batch_embeddings = build_embeddings(
        tokens=batch_tokens, vocab_size=encoder.max_token_value, embed_dim=output_dim, seq_length=seq_length
    )
    print(Fore.CYAN + "Embeddings shape:")
    print(batch_embeddings.shape)  # [8, 4, output_dim]
    print(Fore.CYAN + "Embeddings:")
    print(batch_embeddings)


@app.command()
def qkv(embeddings_dim: int = 3, qkv_dim: int = 2):
    # Using parameters (batch_size=8, stride=1, seq_length=1)
    # for illustration purposes.
    # This results batches of 8 sample, each sample comprised of 4 tokens (seq_length)
    seq_length = 4

    encoder, data_loader = get_encoder_and_batch_iterator(
        Path(__file__).parent.parent.joinpath("resources/the-verdict.txt"),
        batch_size=8,
        stride=4,
        seq_length=seq_length,
    )
    data_iter = iter(data_loader)
    batch_tokens, _ = next(data_iter)

    # 3 dimension word-by-word embedding, dictated by `output_dim=3` and `context_length=seq_length`
    batch_embeddings = build_embeddings(
        tokens=batch_tokens, vocab_size=encoder.max_token_value, embed_dim=embeddings_dim, seq_length=seq_length
    )

    print(Fore.CYAN + "Batch tokens shape:")
    print(f"{batch_tokens.shape}")  # [batch_size, seq_length]
    print(Fore.CYAN + "Embeddings shape:")
    print(batch_embeddings.shape)  # [batch_size, seq_length, output_dim]

    print(Fore.CYAN + ">>> Calculate the qkv vectors for the 2nd element in the batch" + Fore.RESET)
    input_2 = batch_embeddings[1]
    print(f"x_2 shape: {input_2.shape}")  # [seq_length, output_dim]

    print("\n" + Fore.CYAN + ">>> Calculate the full matrices for the batch:" + Fore.RESET)
    # using output_dim=2 as default just for simplicity
    queries, keys, values = build_qkv_matrices(
        batch_embeddings, input_dim=batch_embeddings.shape[2], output_dim=qkv_dim
    )

    print("\n" + Fore.CYAN + "QKV projections:" + Fore.RESET)
    print(Fore.GREEN + f"queries shape: {queries.shape}" + Fore.RESET)  # Shape [batch_size, seq_length, qkv_dim]
    print(f"queries: {queries}")

    print(Fore.GREEN + f"keys shape: {keys.shape}" + Fore.RESET)  # Shape [batch_size, seq_length, qkv_dim]
    print(f"keys: {keys}")

    print(Fore.GREEN + f"values shape: {values.shape}" + Fore.RESET)  # Shape [batch_size, seq_length, qkv_dim]
    print(f"values: {values}")


@app.command()
def attention_for_one(qkv_dim: int = 2):  # noqa: PLR0914
    # Using parameters (batch_size=8, stride=1, seq_length=1)
    # for illustration purposes.
    # This results batches of 8 sample, each sample comprised of 4 tokens (seq_length)
    seq_length = 4
    encoder, data_loader = get_encoder_and_batch_iterator(
        Path(__file__).parent.parent.joinpath("resources/the-verdict.txt"),
        batch_size=8,
        stride=4,
        seq_length=seq_length,
    )
    data_iter = iter(data_loader)
    batch_tokens, _ = next(data_iter)

    batch_embeddings = build_embeddings(
        tokens=batch_tokens, vocab_size=encoder.max_token_value, embed_dim=3, seq_length=seq_length
    )

    print(Fore.CYAN + "Batch tokens shape:")
    print(f"{batch_tokens.shape}")  # [batch_size, seq_length]
    print(Fore.CYAN + "Embeddings shape:")
    print(batch_embeddings.shape)  # [batch_size, seq_length, output_dim]

    queries, keys, values = build_qkv_matrices(
        batch_embeddings, input_dim=batch_embeddings.shape[2], output_dim=qkv_dim
    )

    print()
    print(Fore.CYAN + ">>> Get the 2nd query and key elements" + Fore.RESET)
    query_2, key_2 = queries[1], keys[1]  # pick up one element for illustration

    print(f"query_2 shape: {query_2.shape}")  # Shape [seq_length, qkv_dim]
    print(f"query_2: {query_2}")
    print(f"key_2: {key_2}")

    print(Fore.CYAN + ">>> Compute the attention score for key_2 wrt query_2" + Fore.RESET)
    # Dot product query_2 and each element in key_2.
    attn_score_2 = query_2 @ key_2.T  # [seq_length, seq_length]
    print("unscaled attn_score_2:")
    print(attn_score_2)
    attn_score_22 = torch.dot(query_2[1], key_2[1])
    print(Fore.RED + ">>> sanity test:")
    print(Fore.LIGHTRED_EX + f"{attn_score_22.item():.4f} == {attn_score_2[1][1]:.4f}")

    print(Fore.CYAN + "\n>>> Compute the attention scores for all keys wrt query_2:" + Fore.RESET)
    print(Style.DIM + "Unscaled attention scores..." + Style.NORMAL)
    print(
        Style.DIM
        + "The unscaled attention score is computed as a dot product between the query and the keys vectors."
        + Style.NORMAL
    )
    print(f"keys shape: {keys.shape}")
    # Align the matrices for query_2 and keys
    # query_2 shape is [seq_length, qkv_dim]
    # keys shape is [batch_size, seq_length, qkv_dim]
    # Reshape keys to have the shape [8*4, 2] i.e., combine the batch and token dimension
    keys_reshaped = keys.view(-1, keys.shape[-1])  # shape [32, 2]

    # Now compute the dot product of query_2 with all keys
    # query_2 shape [4, 2], keys_reshaped shape [32, 2]
    attention_scores_2 = query_2 @ keys_reshaped.T  # shape [4, 32]
    print(
        f"attention_scores_2 shape: {attention_scores_2.shape}"
    )  # [8] representing the attention distribution over the 8 elements in the batch.
    print(f"attention_scores_2: \n {attention_scores_2}")

    print()

    print(Fore.CYAN + "\n>>> Compute the attention weights" + Fore.RESET)

    print(Style.DIM + "Scaling the attention weights..." + Style.NORMAL)
    scaling_factor = keys.shape[-1] ** 0.5  # Mathematically equivalent to the square root
    attention_weights_2 = torch.softmax(attention_scores_2 / scaling_factor, dim=-1)
    print(f"attention_weights_2 shape: {attention_weights_2.shape}")
    print(f"attention_weights_2 : \n {attention_weights_2}")

    print()
    print(Fore.CYAN + "\n>>> Compute the context vector" + Fore.RESET)
    # In the attention mechanism, the idea is to weight each value by
    # its corresponding attention weight, then sum over the values to
    # compute the context vector.
    print(
        f"Check values matrix shape: {values.shape}"
        # [8, 1, 2] (for 8 elements, 1 context length, and 2 output dimensions),
        # meaning there are 8 elements, each with a value vector of size 2.
    )

    values_reshaped = values.view(-1, values.shape[-1])  # shape [32, 2]
    # Multiply attention weights with values and sum along the keys dimension
    context_vectors = attention_weights_2 @ values_reshaped  # shape [4, 2]

    print(f"Context vector shape: {context_vectors.shape}")
    print(Fore.GREEN + f"Context vector: {context_vectors}" + Fore.RESET)


def transfer_weights(from_model: SelfAttentionV2, to_model: SelfAttentionV1) -> None:
    # Transfer query weights
    to_model._w_query.data = from_model._w_query.weight.data.T  # Transpose to match the shape  # noqa: SLF001
    # Transfer key weights
    to_model._w_key.data = from_model._w_key.weight.data.T  # Transpose to match the shape  # noqa: SLF001
    # Transfer value weights
    to_model._w_value.data = from_model._w_value.weight.data.T  # Transpose to match the shape  # noqa: SLF001


@app.command()
def attention():
    seq_length = 4
    embedding_dim = 3
    encoder, data_loader = get_encoder_and_batch_iterator(
        Path(__file__).parent.parent.joinpath("resources/the-verdict.txt"),
        batch_size=8,
        stride=4,
        seq_length=seq_length,
    )
    data_iter = iter(data_loader)
    batch_tokens, _ = next(data_iter)

    batch_embeddings = build_embeddings(
        tokens=batch_tokens, vocab_size=encoder.max_token_value, embed_dim=embedding_dim, seq_length=seq_length
    )

    print(Fore.CYAN + "Batch tokens shape:")
    print(f"{batch_tokens.shape}")  # [batch_size, seq_length]
    print(Fore.CYAN + "Embeddings shape:")
    print(batch_embeddings.shape)  # [batch_size, seq_length, output_dim]
    attention_v1 = SelfAttentionV1(embedding_dim=embedding_dim, output_dim=2)
    print(Fore.CYAN + "SelfAttentionV1:" + Fore.RESET)

    print(attention_v1(batch_embeddings))  # implicitly call the forward method

    attention_v2 = SelfAttentionV2(embedding_dim=embedding_dim, output_dim=2)
    print(Fore.GREEN + "SelfAttentionV2:" + Fore.RESET)
    print(attention_v2(batch_embeddings))

    # Verify both are equivalent in working by transferring
    # the weights from attention_v2 to attention_v1
    # and checking if the output is the same for both attention objects
    # Here's the key difference:
    # In SelfAttentionV2, the weights and biases are encapsulated within torch.nn.Linear layers.
    # To access the weights, we use .weight and for the bias (if used), .bias.
    # In SelfAttentionV1, the weights are raw torch.nn.Parameter objects.
    transfer_weights(from_model=attention_v2, to_model=attention_v1)
    print(Fore.YELLOW + "disguised SelfAttentionV2:" + Fore.RESET)
    print(attention_v1(batch_embeddings))


@app.command()
def attention_masked_for_one():
    batch_size = 8
    seq_length = 1
    embedding_dim = 3
    encoder, data_loader = get_encoder_and_batch_iterator(
        Path(__file__).parent.parent.joinpath("resources/the-verdict.txt"),
        batch_size=batch_size,
        stride=1,
        seq_length=seq_length,
    )
    data_iter = iter(data_loader)
    batch_tokens, _ = next(data_iter)

    batch_embeddings = build_embeddings(
        tokens=batch_tokens, vocab_size=encoder.max_token_value, embed_dim=embedding_dim, seq_length=seq_length
    )

    queries, keys, _ = build_qkv_matrices(batch_embeddings, input_dim=embedding_dim, output_dim=2)
    query_2 = queries[1]

    keys_reshaped = keys.view(-1, keys.shape[-1])  # shape [8, 2]

    # Now compute the dot product of query_2 with all keys
    # query_2 shape [1, 2], keys_reshaped shape [8, 2]
    attention_scores = query_2 @ keys_reshaped.T  # shape [1, 8]

    # build a triangular matrix with 0s above the diagonal
    mask = torch.triu(torch.ones(batch_size, batch_size), diagonal=1)
    masked = attention_scores.masked_fill(mask.bool(), -torch.inf)
    attention_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)

    # use an extra dropout layer
    dropout = torch.nn.Dropout(0.2)
    attention_weights = dropout(attention_weights)
    print(Fore.GREEN + "Masked attention for one element:" + Fore.RESET)
    print_attention_matrix(attention_weights)


@app.command()
def attention_masked_for_batch():
    content = load_text_file(Path(__file__).parent.parent.joinpath("resources/the-verdict.txt"))
    encoder = tiktoken.get_encoding("gpt2")
    data_loader = create_dataloader_v1(
        text=content, encoder=encoder, batch_size=2, stride=6, seq_length=6, shuffle=False
    )

    data_iter = iter(data_loader)
    batch_tokens, _ = next(data_iter)
    build_embeddings(tokens=batch_tokens, vocab_size=encoder.max_token_value, embed_dim=3, seq_length=6).squeeze(1)

    # to be continued ...


if __name__ == "__main__":
    app()
