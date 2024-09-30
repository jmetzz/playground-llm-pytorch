"""Command-line interface."""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import tiktoken
import torch
import typer
from colorama import Fore, Style

from nn import SelfAttentionV1, SelfAttentionV2, build_embeddings, build_qkv_matrices
from splitters import punctuation_splitter, space_and_punctuation_splitter, space_splitter
from tokenizers import SimpleRegexTokenizerV1, SimpleRegexTokenizerV2
from utils.data import create_dataloader_v1, load_text_file

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
def dataloader(batch_size: int = 1, stride: int = 1, max_len: int = 4, decode: bool = True):
    the_verdict_file = Path(__file__).parent.parent.joinpath("resources/the-verdict.txt")
    content = load_text_file(the_verdict_file)

    encoder = tiktoken.get_encoding("gpt2")

    dataloader = create_dataloader_v1(
        text=content, encoder=encoder, batch_size=batch_size, stride=stride, max_length=max_len, shuffle=False
    )
    data_iter = iter(dataloader)

    first_batch = next(data_iter)
    output = [[encoder.decode(row.tolist()) for row in tensor] for tensor in first_batch] if decode else first_batch
    print(f"First batch: {output}")

    second_batch = next(data_iter)
    output = [[encoder.decode(row.tolist()) for row in tensor] for tensor in second_batch] if decode else second_batch
    print(f"Second batch: {output}")


def get_encoder_and_batch_iterator(file_path: Path) -> tuple:
    content = load_text_file(file_path)
    encoder = tiktoken.get_encoding("gpt2")
    dataloader = create_dataloader_v1(
        text=content, encoder=encoder, batch_size=8, stride=1, max_length=1, shuffle=False
    )
    return encoder, iter(dataloader)


@app.command()
def embedding(
    batch_size: int = 8,
    stride: int = 4,
    max_len: int = 4,
    output_dim: int = 256,
):
    the_verdict_file = Path(__file__).parent.parent.joinpath("resources/the-verdict.txt")
    content = load_text_file(the_verdict_file)

    encoder = tiktoken.get_encoding("gpt2")
    dataloader = create_dataloader_v1(
        text=content, encoder=encoder, batch_size=batch_size, stride=stride, max_length=max_len, shuffle=False
    )
    data_iter = iter(dataloader)

    input_tokens, _ = next(data_iter)
    print(f"Token IDs: {input_tokens}")
    print(f"Inputs shape: {input_tokens.shape}")  # [8, 4]

    input_embeddings = build_embeddings(
        inputs=input_tokens, vocab_size=encoder.max_token_value, output_dim=output_dim, context_length=max_len
    )
    print(input_embeddings.shape)  # [8, 4, 256]


@app.command()
def qkv():
    the_verdict_file = Path(__file__).parent.parent.joinpath("resources/the-verdict.txt")
    # with default parameters for illustration purposes:
    #     a batch of 8 x 1D sample (batch_size=8, stride=1, max_length=1)
    encoder, data_iter = get_encoder_and_batch_iterator(the_verdict_file)

    input_tokens, _ = next(data_iter)
    print(f"Batch inputs shape: {input_tokens.shape}")  # [8, 1]
    print(f"Token IDs: {input_tokens}")

    # 3 dimension word-by-word embedding, dictated by `output_dim=3` and `context_length=1`
    input_embeddings = build_embeddings(
        inputs=input_tokens, vocab_size=encoder.max_token_value, output_dim=3, context_length=1
    )

    print(
        f"Inputs embeddings shape: {input_embeddings.shape}"
    )  # [8, 1, 3] (batch_size=8, context_length=1, output_dim=3)
    print(f"Inputs embeddings:\n{input_embeddings}")

    print(Fore.CYAN + ">>> Calculate the qkv vectors for the second element in the batch" + Fore.RESET)
    input_2 = input_embeddings[1]
    print(f"x_2 shape: {input_2.shape}")  # [1, 3] (context_length=1, output_dim=3)

    print("\n" + Fore.CYAN + ">>> Calculate the full matrices for the batch:" + Fore.RESET)
    # using output_dim=2 just for simplicity
    queries, keys, values = build_qkv_matrices(input_embeddings, input_dim=input_embeddings.shape[2], output_dim=2)

    print("\n" + Fore.CYAN + "Projections:" + Fore.RESET)
    print(Fore.GREEN + f"queries shape: {queries.shape}" + Fore.RESET)  # Shape [8, 1, 2]
    print(f"queries: {queries}")

    print(Fore.GREEN + f"keys shape: {keys.shape}" + Fore.RESET)  # Shape [8, 1, 2]
    print(f"keys: {keys}")

    print(Fore.GREEN + f"values shape: {values.shape}" + Fore.RESET)  # Shape [8, 1, 2]
    print(f"values: {values}")


@app.command()
def attention_for_one_element():  # noqa: PLR0914
    the_verdict_file = Path(__file__).parent.parent.joinpath("resources/the-verdict.txt")
    # with default parameters for illustration purposes:
    #     a batch of 8 x 1D sample (batch_size=8, stride=1, max_length=1)
    encoder, data_iter = get_encoder_and_batch_iterator(the_verdict_file)
    input_tokens, _ = next(data_iter)
    input_embeddings = build_embeddings(
        inputs=input_tokens, vocab_size=encoder.max_token_value, output_dim=3, context_length=1
    ).squeeze(1)

    queries, keys, values = build_qkv_matrices(input_embeddings, input_dim=input_embeddings.shape[1], output_dim=2)

    query_2 = queries[1]
    key_2 = keys[1]

    print(Fore.CYAN + ">>> Get the second input element (input_2)" + Fore.RESET)
    print(f"\nquery_2 shape: {query_2.shape}")  # Shape [2]
    print(f"query_2: {query_2}")
    print(f"key_2: {key_2}")

    print(Fore.CYAN + ">>> Compute the attention score for key_2 wrt query_2" + Fore.RESET)
    # Dot product between query_2 and key_2.
    attn_score_22 = torch.dot(query_2, key_2)
    print(f"unscaled attn_score_22: {attn_score_22:.4f}")

    print(Fore.CYAN + "\n>>> Compute the attention scores for all keys wrt query_2:" + Fore.RESET)
    print(Style.DIM + "Unscaled attention scores..." + Style.NORMAL)
    print(
        Style.DIM
        + "The unscaled attention score is computed as a dot product between the query and the keys vectors."
        + Style.NORMAL
    )
    print(f"keys shape: {keys.shape}")
    # Align the matrices for query_2 and keys
    attention_scores_2 = query_2 @ keys.permute(1, 0)
    # print(f"attention_scores_2 shape after @ operation: {attention_scores_2.shape}")
    # print("Squeezing it ...")
    # attention_scores_2 = attention_scores_2.squeeze(-1)
    print(
        f"attention_scores_2 shape: {attention_scores_2.shape}"
    )  # [8] representing the attention distribution over the 8 elements in the batch.
    print(f"attention_scores_2: \n {attention_scores_2}")

    # sanity check
    print(
        Style.DIM
        + Fore.RED
        + f"\t[Sanity check] {attn_score_22:.5f} == {attention_scores_2[1].item():.5f}"
        + Fore.RESET
        + Style.NORMAL
    )
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

    # Firstly, reshape attention_weights_2 to enable proper broadcastingShape:
    attention_weights_2 = attention_weights_2.unsqueeze(1)
    print(f"attention_weights_2 reshaped: {attention_weights_2.shape}")  # new shape [8, 1]

    context_vectors = torch.sum(attention_weights_2 * values, dim=0)

    print()
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
    the_verdict_file = Path(__file__).parent.parent.joinpath("resources/the-verdict.txt")
    encoder, data_iter = get_encoder_and_batch_iterator(the_verdict_file)
    input_tokens, _ = next(data_iter)
    input_embeddings = build_embeddings(
        inputs=input_tokens, vocab_size=encoder.max_token_value, output_dim=3, context_length=1
    ).squeeze(1)

    attention_v1 = SelfAttentionV1(input_dim=input_embeddings.shape[1], output_dim=2)
    print(Fore.CYAN + "SelfAttentionV1:" + Fore.RESET)
    print(attention_v1(input_embeddings))  # implicitly call the forward method

    attention_v2 = SelfAttentionV2(input_dim=input_embeddings.shape[1], output_dim=2)
    print(Fore.GREEN + "SelfAttentionV2:" + Fore.RESET)
    print(attention_v2(input_embeddings))

    # Verify both are equivalent in working by transferring
    # the weights from attention_v2 to attention_v1
    # and checking if the output is the same for both attention objects
    # Here's the key difference:
    # In SelfAttentionV2, the weights and biases are encapsulated within torch.nn.Linear layers.
    # To access the weights, we use .weight and for the bias (if used), .bias.
    # In SelfAttentionV1, the weights are raw torch.nn.Parameter objects.
    transfer_weights(from_model=attention_v2, to_model=attention_v1)
    print(Fore.YELLOW + "disguised SelfAttentionV2:" + Fore.RESET)
    print(attention_v1(input_embeddings))


if __name__ == "__main__":
    app()
