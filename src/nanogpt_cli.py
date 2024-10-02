"""Command-line interface."""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import colorama
import torch
import typer
from colorama import Fore, Style

from common.data import get_encoder_and_batch_iterator, load_text_file
from common.io import print_attention_matrix
from nano_gpt.nn import (
    CausalAttentionV1,
    MultiHeadAttentionV1,
    SelfAttentionV1,
    SelfAttentionV2,
    build_embeddings,
    build_qkv_matrices,
)
from nano_gpt.splitters import punctuation_splitter, space_and_punctuation_splitter, space_splitter
from nano_gpt.tokenizers import SimpleRegexTokenizerV1, SimpleRegexTokenizerV2

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

    print(f"{Fore.CYAN}Token IDs:")
    print(f"Batch inputs shape: {batch_tokens.shape}")  # [8, 1]
    print(f"{batch_tokens}")

    print("f{Fore.YELLOW}Label IDs:")
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
    print(f"{Fore.CYAN}>>> Testing data loader with parameters:")
    print(f"{Fore.CYAN}\t batch_size: 1")
    print(f"{Fore.CYAN}\t stride: 1")
    print(f"{Fore.CYAN}\t seq_length: 4")

    for idx in range(2):
        batch_tokens, batch_labels = next(data_iter)
        batch_tokens = batch_tokens.squeeze()
        batch_labels = batch_labels.squeeze()

        print(f"{Fore.YELLOW}>>> Batch #{idx}")
        print(f"{Fore.LIGHTCYAN_EX}encoded tokens{Fore.RESET}: {batch_tokens}")
        print(f"{Fore.LIGHTCYAN_EX}encoded label{Fore.RESET}: {batch_labels}")

        output = [encoder.decode([element]) for element in batch_tokens]
        print(f"{Fore.BLUE}decoded tokens{Fore.RESET}: {output}")

        output = [encoder.decode([element]) for element in batch_labels]
        print(f"{Fore.BLUE}decoded label{Fore.RESET}: {output}")


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
    print(f"{Fore.CYAN}Batch tokens shape:")
    print(f"{batch_tokens.shape}")  # [batch_size, seq_length]
    print(f"{Fore.CYAN}Token IDs:")
    print(f"{batch_tokens}")

    batch_embeddings = build_embeddings(
        tokens=batch_tokens, vocab_size=encoder.max_token_value, embeddings_dim=output_dim, seq_length=seq_length
    )
    print(f"{Fore.CYAN}Embeddings shape:")
    print(batch_embeddings.shape)  # [8, 4, output_dim]
    print(f"{Fore.CYAN}Embeddings:")
    print(batch_embeddings)


@app.command()
def qkv(batch_size: int = 8, seq_length: int = 4, stride: int = 4, embeddings_dim: int = 3, qkv_dim: int = 2):
    # Using parameters (batch_size=8, stride=1, seq_length=1)
    # for illustration purposes.
    # This results batches of 8 sample, each sample comprised of 4 tokens (seq_length)

    encoder, data_loader = get_encoder_and_batch_iterator(
        Path(__file__).parent.parent.joinpath("resources/the-verdict.txt"),
        batch_size=batch_size,
        stride=stride,
        seq_length=seq_length,
    )
    data_iter = iter(data_loader)
    batch_tokens, _ = next(data_iter)

    # 3 dimension word-by-word embedding, dictated by `output_dim=3` and `context_length=seq_length`
    batch_embeddings = build_embeddings(
        tokens=batch_tokens, vocab_size=encoder.max_token_value, embeddings_dim=embeddings_dim, seq_length=seq_length
    )

    print(f"{Fore.CYAN}Batch tokens shape:")
    print(f"{batch_tokens.shape}")  # [batch_size, seq_length]
    print(f"{Fore.CYAN}Embeddings shape:")
    print(batch_embeddings.shape)  # [batch_size, seq_length, output_dim]

    print(f"{Fore.CYAN}>>> Calculate the qkv vectors for the 2nd element in the batch")
    input_2 = batch_embeddings[1]
    print(f"x_2 shape: {input_2.shape}")  # [seq_length, output_dim]

    print(f"{Fore.CYAN}\n>>> Calculate the full matrices for the batch:")
    queries, keys, values = build_qkv_matrices(batch_embeddings, embeddings_dim=embeddings_dim, qkv_dim=qkv_dim)

    print(f"{Fore.CYAN}QKV projections:")
    print(f"{Fore.GREEN}queries shape: {queries.shape}")  # Shape [batch_size, seq_length, qkv_dim]
    print(f"queries: {queries}")

    print(f"{Fore.GREEN}keys shape: {keys.shape}")  # Shape [batch_size, seq_length, qkv_dim]
    print(f"keys: {keys}")

    print(f"{Fore.GREEN}values shape: {values.shape}")  # Shape [batch_size, seq_length, qkv_dim]
    print(f"values: {values}")


@app.command()
def attention_for_one_query(  # noqa: PLR0913, PLR0914
    batch_size: int = 8,
    seq_length: int = 4,
    stride: int = 4,
    embeddings_dim: int = 3,
    qkv_dim: int = 2,
    *,
    verbose: bool = False,
):
    encoder, data_loader = get_encoder_and_batch_iterator(
        Path(__file__).parent.parent.joinpath("resources/the-verdict.txt"),
        batch_size=batch_size,
        stride=stride,
        seq_length=seq_length,
    )
    data_iter = iter(data_loader)
    batch_tokens, _ = next(data_iter)

    batch_embeddings = build_embeddings(
        tokens=batch_tokens, vocab_size=encoder.max_token_value, embeddings_dim=embeddings_dim, seq_length=seq_length
    )

    queries, keys, values = build_qkv_matrices(batch_embeddings, embeddings_dim=embeddings_dim, qkv_dim=qkv_dim)
    query_last_token, key_last = queries[-1], keys[-1]  # pick up the last element for illustration

    # Align the matrices for query and keys
    # query shape is [seq_length, qkv_dim]
    # keys shape is [batch_size, seq_length, qkv_dim]
    # Reshape keys to have the shape [batch_size * seq_length, qk   v_dim] i.e., combine the batch and token dimension
    keys_reshaped = keys.view(-1, keys.shape[-1])

    # Now compute the dot product of query with all keys
    # query shape [seq_length, qkv_dim], keys_reshaped shape [batch_size * seq_length, qkv_dim]
    # attention scores tell you how much focus it should give to other tokens in the same sequence
    attention_scores = query_last_token @ keys_reshaped.T
    scaling_factor = keys.shape[-1] ** 0.5  # Mathematically equivalent to the square root
    attention_weights = torch.softmax(attention_scores / scaling_factor, dim=-1)

    # In the attention mechanism, the idea is to weight each value by
    # its corresponding attention weight, then sum over the values to
    # compute the context vector.
    values_reshaped = values.view(-1, values.shape[-1])  # shape [batch_size * seq_length, qkv_dim]
    # Multiply attention weights with values and sum along the keys dimension
    context_vectors = attention_weights @ values_reshaped  # shape [seq_length, qkv_dim]

    # --- Prepare and print the output information ---
    print(f"{Fore.CYAN}batch_tokens{Fore.RESET}: {batch_tokens.shape}")  # [batch_size, seq_length]
    print(f"{Fore.CYAN}batch_embeddings{Fore.RESET}: {batch_embeddings.shape}")  # [batch_size, seq_length, output_dim]
    print(f"{Fore.CYAN}queries{Fore.RESET}: {queries.shape}")  # shape [batch_size, seq_length, qkv_dim]
    print(f"{Fore.CYAN}keys{Fore.RESET}: {keys.shape}")  # shape [batch_size, seq_length, qkv_dim]
    print(f"{Fore.CYAN}keys reshaped{Fore.RESET}: {keys_reshaped.shape}")
    print(f"{Fore.CYAN}values{Fore.RESET} {values.shape}")

    print(f"{Fore.CYAN}query{Fore.RESET}: {query_last_token.shape}")  # Shape [seq_length, qkv_dim]
    print(f"{Fore.CYAN}attention_scores{Fore.RESET}: {attention_scores.shape}")  # [batch_size, seq_length, seq_length]
    print(
        f"{Fore.CYAN}attention_weights{Fore.RESET}: {attention_weights.shape}"
    )  # [batch_size, seq_length, seq_length]
    print(f"{Fore.CYAN}context_vectors{Fore.RESET}: {context_vectors.shape}")

    if verbose:
        print(f"{Fore.CYAN}\n>>> query and key elements")
        print(f"query: {query_last_token}")
        print(f"key: {key_last}")

        print(Fore.CYAN + "\n>>> Computed the attention scores for all keys wrt query:")
        print(
            Style.DIM + "Unscaled attention score is computed as a dot product between the query and the keys vectors."
        )
        print(f"attention_scores: \n {attention_scores}")

        print(f"{Fore.LIGHTRED_EX}\t ~~~ sanity test:")
        attn_score_dot = query_last_token @ key_last.T  # [1, seq_length]
        if seq_length > 1:
            # choose second token if possible
            attn_score_2 = torch.dot(query_last_token[1], key_last[1])
            print(Fore.LIGHTRED_EX + f"\t {attn_score_2.item():.4f} == {attn_score_dot[1][1].item():.4f}")
        else:
            attn_score_2 = torch.dot(query_last_token.flatten(), key_last.flatten())
            print(Fore.LIGHTRED_EX + f"\t {attn_score_2.item():.4f} == {attn_score_dot.flatten().item():.4f}")
        print(f"{Fore.LIGHTRED_EX}\t unscaled attn_score{Fore.RESET}: {attn_score_dot}")

        print(Fore.CYAN + "\n>>> Computed attention weights")
        print_attention_matrix(attention_weights)

        print(f"\n{Fore.CYAN}>>> Computed the context vector")
        print(Fore.GREEN + f"Context vector: {context_vectors}")


@app.command()
def attention_simple(
    batch_size: int = 2, seq_length: int = 1, stride: int = 1, embeddings_dim: int = 3, qkv_dim: int = 2
):
    encoder, data_loader = get_encoder_and_batch_iterator(
        Path(__file__).parent.parent.joinpath("resources/the-verdict.txt"),
        batch_size=batch_size,
        stride=stride,
        seq_length=seq_length,
    )
    data_iter = iter(data_loader)
    batch_tokens, _ = next(data_iter)

    batch_embeddings = build_embeddings(
        tokens=batch_tokens, vocab_size=encoder.max_token_value, embeddings_dim=embeddings_dim, seq_length=seq_length
    )

    print(f"{Fore.CYAN}Batch tokens shape:")
    print(f"{batch_tokens.shape}")  # [batch_size, seq_length]
    print(f"{Fore.CYAN}Embeddings shape:")
    print(batch_embeddings.shape)  # [batch_size, seq_length, output_dim]
    attention_v1 = SelfAttentionV1(embeddings_dim=embeddings_dim, output_dim=qkv_dim)
    print(f"{Fore.CYAN}SelfAttentionV1:")

    print(attention_v1(batch_embeddings))  # implicitly call the forward method

    attention_v2 = SelfAttentionV2(embeddings_dim=embeddings_dim, output_dim=qkv_dim)
    print(f"{Fore.CYAN}SelfAttentionV2:")
    print(attention_v2(batch_embeddings))

    # Verify both are equivalent in working by transferring
    # the weights from attention_v2 to attention_v1
    # and checking if the output is the same for both attention objects
    # Here's the key difference:
    # In SelfAttentionV2, the weights and biases are encapsulated within torch.nn.Linear layers.
    # To access the weights, we use .weight and for the bias (if used), .bias.
    # In SelfAttentionV1, the weights are raw torch.nn.Parameter objects.
    def _transfer_weights(from_model: SelfAttentionV2, to_model: SelfAttentionV1) -> None:
        # Transfer query weights
        to_model._w_query.data = from_model._w_query.weight.data.T  # Transpose to match the shape  # noqa: SLF001
        # Transfer key weights
        to_model._w_key.data = from_model._w_key.weight.data.T  # Transpose to match the shape  # noqa: SLF001
        # Transfer value weights
        to_model._w_value.data = from_model._w_value.weight.data.T  # Transpose to match the shape  # noqa: SLF001

    _transfer_weights(from_model=attention_v2, to_model=attention_v1)
    print(f"{Fore.YELLOW}disguised SelfAttentionV2:")
    print(attention_v1(batch_embeddings))


@app.command()
def attention_masked(  # noqa: PLR0913, PLR0914, PLR0917
    batch_size: int = 8,
    stride: int = 4,
    seq_length: int = 4,
    embeddings_dim: int = 3,
    qkv_dim: int = 2,
    dropout: float | None = 0.2,
    for_item: int | None = None,
    *,
    verbose: bool = False,
):
    encoder, data_loader = get_encoder_and_batch_iterator(
        Path(__file__).parent.parent.joinpath("resources/the-verdict.txt"),
        batch_size=batch_size,
        stride=stride,
        seq_length=seq_length,
    )
    data_iter = iter(data_loader)
    batch_tokens, _ = next(data_iter)

    batch_embeddings = build_embeddings(
        tokens=batch_tokens, vocab_size=encoder.max_token_value, embeddings_dim=embeddings_dim, seq_length=seq_length
    )
    queries, keys, values = build_qkv_matrices(batch_embeddings, embeddings_dim=embeddings_dim, qkv_dim=qkv_dim)

    if for_item:
        # Compute the dot product of query_item with all keys
        # query_item shape [1, qvk_output_dim], keys_reshaped shape [batch_size, qkv_output_dim]
        query_item = queries[for_item]
        keys_reshaped = keys.view(-1, keys.shape[-1])  # shape [batch_size, qkv_output_dim]

        # Now compute the dot product of query_2 with all keys
        # query_2 shape [1, qkv_output_dim], keys_reshaped shape [batch_size, qkv_output_dim]
        attention_scores = query_item @ keys_reshaped.T  # shape [1, batch_size]
    else:  # for batch
        # Compute attention scores for all elements in the batch
        # via batch matrix-matrix product of matrices queries and keys.
        # - queries shape: [batch_size, seq_length, qkv_output_dim],
        #   keys shape: [batch_size, seq_length, qkv_output_dim]
        # - for each query, compute a dot product with every key,
        #   resulting in an attention score matrix of size
        #   seq_length x seq_length for each batch element.
        attention_scores = torch.bmm(queries, keys.transpose(-2, -1))  # shape: [batch_size, seq_length, seq_length]

    # Apply mask (triangular matrix to mask future positions)
    # A mask is used to prevent attention to future positions in the sequence
    # (i.e., positions that are to the right of the current position)
    # ### mask = torch.triu(torch.ones(batch_size, batch_size), diagonal=1)
    # In self-attention, you are masking positions within a sequence, not across different batches.
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    # Then, you would need to broadcast the mask across the batch dimension:
    mask.unsqueeze(0).expand(batch_size, -1, -1)
    masked_attention_scores = attention_scores.masked_fill(mask.bool(), -torch.inf)

    # Apply softmax to normalized scores
    scaling_factor = keys.shape[-1] ** 0.5  # Mathematically equivalent to the square root
    attention_weights = torch.softmax(masked_attention_scores / scaling_factor, dim=-1)

    # use an extra dropout layer
    if dropout:
        attention_weights = torch.nn.Dropout(dropout)(attention_weights)

    context_vectors = torch.bmm(attention_weights, values)  # shape [batch_size, seq_length, qkv_dim]

    print(f"{Fore.CYAN}batch_tokens{Fore.RESET}: {batch_tokens.shape}")  # [batch_size, seq_length]
    print(f"{Fore.CYAN}batch_embeddings{Fore.RESET}: {batch_embeddings.shape}")  # [batch_size, seq_length, output_dim]
    print(f"{Fore.CYAN}queries{Fore.RESET}: {queries.shape}")  # shape [batch_size, seq_length, qkv_dim]
    print(f"{Fore.CYAN}keys{Fore.RESET}: {keys.shape}")  # shape [batch_size, seq_length, qkv_dim]
    print(f"{Fore.CYAN}values{Fore.RESET} {values.shape}")
    print(f"{Fore.CYAN}attention_scores{Fore.RESET}: {attention_scores.shape}")  # [batch_size, seq_length, seq_length]
    print(
        f"{Fore.CYAN}masked_attention_scores{Fore.RESET}: {masked_attention_scores.shape}"
    )  # [batch_size, seq_length, seq_length]
    print(
        f"{Fore.CYAN}attention_weights{Fore.RESET}: {attention_weights.shape}"
    )  # [batch_size, seq_length, seq_length]
    print(f"{Fore.CYAN}context_vectors{Fore.RESET}: {context_vectors.shape}")

    if verbose:
        print(f"{Fore.CYAN}Masked attention matrices:")
        if for_item:
            print_attention_matrix(attention_weights)
        else:
            for idx in range(attention_weights.shape[0]):
                print_attention_matrix(attention_weights[idx])
                print()

        print(Fore.GREEN + "Context vector:")
        print(context_vectors)


@app.command()
def attention_causal(
    batch_size: int = 8, stride: int = 4, seq_length: int = 4, embeddings_dim: int = 3, qkv_dim: int = 2
):
    encoder, data_loader = get_encoder_and_batch_iterator(
        Path(__file__).parent.parent.joinpath("resources/the-verdict.txt"),
        batch_size=batch_size,
        stride=stride,
        seq_length=seq_length,
    )
    data_iter = iter(data_loader)
    batch_tokens, _ = next(data_iter)

    batch_embeddings = build_embeddings(
        tokens=batch_tokens, vocab_size=encoder.max_token_value, embeddings_dim=embeddings_dim, seq_length=seq_length
    )

    causal_attention = CausalAttentionV1(embedding_dim=embeddings_dim, output_dim=qkv_dim, dropout=0.0)
    context_vectors = causal_attention(batch_embeddings)

    print(f"{Fore.CYAN}batch_embeddings{Fore.RESET}: {batch_embeddings.shape}")  # [batch_size, seq_length, output_dim]
    print(f"{Fore.CYAN}batch_tokens{Fore.RESET}: {batch_tokens.shape}")  # [batch_size, seq_length]
    print(f"{Fore.CYAN}context_vectors{Fore.RESET}: {context_vectors.shape}")


@app.command()
def attention_multihead(
    batch_size: int = 8, stride: int = 4, seq_length: int = 4, embeddings_dim: int = 8, num_heads: int = 2
):
    encoder, data_loader = get_encoder_and_batch_iterator(
        Path(__file__).parent.parent.joinpath("resources/the-verdict.txt"),
        batch_size=batch_size,
        stride=stride,
        seq_length=seq_length,
    )
    data_iter = iter(data_loader)
    batch_tokens, _ = next(data_iter)

    batch_embeddings = build_embeddings(
        tokens=batch_tokens, vocab_size=encoder.max_token_value, embeddings_dim=embeddings_dim, seq_length=seq_length
    )

    # how to define the number of heads and head size?
    # - head_dim = embeddings_dim // num_heads
    # since each head will result tensors with head_dim dimension,
    # and then, when concatenated, the overall result will have the same dimension size
    # as the input (embedding_dim). This simulates a group convolution behavior.
    multihead = MultiHeadAttentionV1(
        embeddings_dim=embeddings_dim, head_dim=embeddings_dim // num_heads, num_heads=num_heads
    )
    context_vectors = multihead(batch_embeddings)

    # [batch_size, seq_length, embeddings_dim]
    print(f"{Fore.CYAN}batch_embeddings{Fore.RESET}: {batch_embeddings.shape}")

    # [batch_size, seq_length]
    print(f"{Fore.CYAN}batch_tokens{Fore.RESET}: {batch_tokens.shape}")

    # [batch_size, seq_length, embeddings_dim]
    print(f"{Fore.CYAN}context_vectors{Fore.RESET}: {context_vectors.shape}")


if __name__ == "__main__":
    app()
