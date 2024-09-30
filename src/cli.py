"""Command-line interface."""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import tiktoken
import torch
import typer
from colorama import Fore, Style

from playground_llm import build_embeddings, build_qkv_matrices
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
def qkv_matrices():
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
    print(f"queries shape: {queries.shape}")  # Shape [8, 1, 2]
    print(f"queries: \n{queries}")

    print(f"keys shape: {keys.shape}")  # Shape [8, 1, 2]
    print(f"keys: \n{keys}")

    print(f"values shape: {values.shape}")  # Shape [8, 1, 2]
    print(f"values: \n{values}")



if __name__ == "__main__":
    app()
