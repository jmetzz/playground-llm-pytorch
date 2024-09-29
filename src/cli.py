"""Command-line interface."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import tiktoken
import torch
import typer

from playground_llm import build_embeddings
from splitters import punctuation_splitter, space_and_punctuation_splitter, space_splitter
from tokenizers import SimpleRegexTokenizerV1, SimpleRegexTokenizerV2
from utils.data import create_dataloader_v1, load_text_file

app = typer.Typer()


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
def qkv_matrices():  # noqa: PLR0914
    the_verdict_file = Path(__file__).parent.parent.joinpath("resources/the-verdict.txt")
    content = load_text_file(the_verdict_file)

    encoder = tiktoken.get_encoding("gpt2")
    dataloader = create_dataloader_v1(
        text=content, encoder=encoder, batch_size=8, stride=1, max_length=1, shuffle=False
    )
    data_iter = iter(dataloader)

    input_tokens, _ = next(data_iter)
    print(f"Batch inputs shape: {input_tokens.shape}")  # [8, 1]
    print(f"Token IDs: {input_tokens}")

    # 3 dimension word-by-word embedding, dictated by `output_dim=3` and `context_length=1`
    input_embeddings = build_embeddings(
        inputs=input_tokens, vocab_size=encoder.max_token_value, output_dim=3, context_length=1
    )
    print(f"Inputs embeddings shape: {input_embeddings.shape}")  # [8, 1, 3]
    print(f"Inputs embeddings:\n{input_embeddings}")

    print(">>> Calculate the qkv vectors for the second element in the batch")
    input_2 = input_embeddings[1]
    print(f"x_2 shape: {input_2.shape}")  # [1, 3]

    torch.manual_seed(123)
    _, d_in = input_2.shape
    d_out = 2  # using output 2 just for simplicity

    # set `requires_grad=True` to update these matrices during model training
    w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
    w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
    w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)

    print(f"W_q shape: {w_query.shape}")  # Shape [3, 2]
    print(f"W_k shape: {w_key.shape}")  # Shape [3, 2]
    print(f"W_v shape: {w_value.shape}")  # Shape [3, 2]

    query_2 = input_2 @ w_query
    key_2 = input_2 @ w_key
    value_2 = input_2 @ w_value

    print(f"\nquery_2 shape: {query_2.shape}")  # Shape [1, 2]
    print(f"query_2: {query_2}")
    print(f"key_2: {key_2}")
    print(f"value_2: {value_2}\n")

    print("\n>>> Calculate the full matrices for the batch:")
    queries = input_embeddings @ w_query
    keys = input_embeddings @ w_key
    values = input_embeddings @ w_value

    print("\n>>> Projections:")
    print(f"queries: \n{queries}")
    print(f"keys: \n{keys}")
    print(f"values: \n{values}")
    print(f"queries shape: {queries.shape}")  # Shape [8, 1, 2]
    print(f"keys shape: {keys.shape}")  # Shape [8, 1, 2]
    print(f"values shape: {values.shape}")  # Shape [8, 1, 2]


if __name__ == "__main__":
    app()
