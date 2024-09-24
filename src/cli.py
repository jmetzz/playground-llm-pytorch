"""Command-line interface."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from tokenizers import SimpleRegexTokenizerV1, punctuation_splitter, space_and_punctuation_splitter, space_splitter
from utils import load_text_file

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
def tokenize():
    the_verdict_file = Path(__file__).parent.parent.joinpath("resources/the-verdict.txt")
    content = load_text_file(the_verdict_file)
    words = punctuation_splitter(content, encode_spaces=False)
    print(f"number of words: {len(words)}")

    vocabulary = {token: idx for idx, token in enumerate(words)}
    print(f"vocabulary size: {len(vocabulary)}")

    tokenizer = SimpleRegexTokenizerV1(vocabulary)

    text = """\"It's the last he painted, you know,\"
       Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    app()
