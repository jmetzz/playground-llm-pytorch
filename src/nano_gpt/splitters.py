import re

SPACE_SPLITTER = re.compile(r"(\s)")
PUNCTUATION_SPLITTER = re.compile(r"([-.,;:?!'\"\[\]\(\)])")
SPACE_AND_PUNCTUATION_SPLITTER = re.compile(r"([-.,;:?!'\"\[\]\(\)\s])")


def _regex_splitter(pattern: re.Pattern[str], text: str, *, encode_spaces: bool) -> list[str]:
    tokens = pattern.split(text)
    if encode_spaces:
        return [token for token in tokens if token]
    # Remove empty strings
    return [token for token in tokens if token.strip()]


def space_splitter(text: str, *, encode_spaces: bool) -> list[str]:
    """Splits text based on spaces, optionally encoding spaces as separate tokens.

    Args:
        text (str): The input text to be split.
        encode_spaces (bool): If True, includes spaces as separate tokens.

    Returns:
        list[str]: A list of tokens split by spaces. Spaces are included if `encode_space` is True.
    """
    return _regex_splitter(SPACE_SPLITTER, text, encode_spaces=encode_spaces)


def punctuation_splitter(text: str, *, encode_spaces: bool) -> list[str]:
    """Splits text into words and punctuation marks, optionally encoding spaces as separate tokens.

    Args:
        text (str): The input text to be split.
        encode_spaces (bool): If True, includes spaces as separate tokens.

    Returns:
        list[str]: A list of tokens including words and punctuation. Spaces are included if `encode_space` is True.
    """
    return _regex_splitter(PUNCTUATION_SPLITTER, text, encode_spaces=encode_spaces)


def space_and_punctuation_splitter(text: str, *, encode_spaces: bool) -> list[str]:
    """Splits text into words and punctuation marks, optionally encoding spaces as separate tokens.

    Args:
        text (str): The input text to be split.
        encode_spaces (bool): If True, includes spaces as separate tokens.

    Returns:
        list[str]: A list of tokens including words and punctuation. Spaces are included if `encode_space` is True.
    """
    return _regex_splitter(SPACE_AND_PUNCTUATION_SPLITTER, text, encode_spaces=encode_spaces)
