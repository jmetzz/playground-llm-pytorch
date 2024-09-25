import re


class SimpleRegexTokenizerV1:
    """A simple tokenizer that splits text based on a regular expression and provides methods for encoding and decoding.

    Args:
        vocabulary (dict[str, int]): A dictionary mapping tokens (words) to unique integer IDs.
        regex (str | None, optional): A regular expression for tokenizing the text. Defaults to a punctuation splitter.
    """

    def __init__(self, vocabulary: list[str], regex: str | None = None) -> None:
        words = sorted(list(set(vocabulary)))
        self._word_to_index = {token: idx for idx, token in enumerate(words)}
        self._index_to_word = {idx: w for w, idx in self._word_to_index.items()}
        regex = regex or r"([-.,;:?!'\"\[\]\(\)\s]|\w+)"
        self._matcher = re.compile(regex)

    def encode(self, text: str) -> list[int]:
        """Encodes the input text as a list of integer IDs based on the vocabulary.

        Args:
            text (str): The input text to be encoded.

        Returns:
            list[int]: A list of integer IDs corresponding to the tokens in the text.

        Raises:
            ValueError: If a token is not found in the vocabulary.
        """
        # Match words and punctuation as separate tokens
        tokens = self._matcher.findall(text)
        tokens = [token for token in tokens if token.strip()]
        try:
            return [self._word_to_index[token] for token in tokens]
        except KeyError as err:
            raise ValueError(f"Token '{err.args[0]}' not found in vocabulary.") from err

    def decode(self, ids: list[int], sep: str = " ") -> str:
        """Decodes a list of integer IDs back into a string using the vocabulary.

        Args:
            ids (list[int]): A list of integer IDs to be decoded.
            sep (str, optional): The separator used to join the decoded tokens. Defaults to a single space.

        Returns:
            str: The decoded string from the list of integer IDs.

        Raises:
            ValueError: If an ID is not found in the reverse vocabulary.
        """
        try:
            words = [self._index_to_word[identifier] for identifier in ids]
        except KeyError as err:
            raise ValueError(f"Identifier '{err.args[0]}' not found in reverse vocabulary.") from err

        text = sep.join(words)
        # Remove spaces before punctuation
        return re.sub(r"\s+([-.,;:?!'\"\)\]])", r"\1", text)


class SimpleRegexTokenizerV2:
    """A simple tokenizer that splits text based on a regular expression and
    provides methods for encoding and decoding.

    This tokenizer includes two special token: "<|unk|>" and "<|endoftext|>"
    in order to handle unknown words and understanding of context.

    Args:
        vocabulary (dict[str, int]): A dictionary mapping tokens (words) to unique integer IDs.
        regex (str | None, optional): A regular expression for tokenizing the text. Defaults to a punctuation splitter.
    """

    BLOCK_DELIMITER = "<|endoftext|>"
    UNKNOWN_TOKEN = "<|unk|>"  # noqa: S105

    def __init__(
        self, vocabulary: list[str], especial_tokens: list[str] | None = None, regex: str | None = None
    ) -> None:
        words = sorted(list(set([*vocabulary, self.BLOCK_DELIMITER, self.UNKNOWN_TOKEN])))
        if especial_tokens:
            words.update(*especial_tokens)

        self._word_to_index = {token: idx for idx, token in enumerate(words)}
        self._index_to_word = {idx: w for w, idx in self._word_to_index.items()}
        regex = regex or r"([-.,;:?!'\"\[\]\(\)\s]|\w+)"
        self._matcher = re.compile(regex)
        self._UNKNOWN_TOKEN_ID = self._word_to_index[self.UNKNOWN_TOKEN]

    def encode(self, text: str) -> list[int]:
        """Encodes the input text as a list of integer IDs based on the vocabulary.

        Args:
            text (str): The input text to be encoded.

        Returns:
            list[int]: A list of integer IDs corresponding to the tokens in the text.
        """
        # Match words and punctuation as separate tokens
        tokens = self._matcher.findall(text)
        tokens = [token.strip() for token in tokens if token.strip()]
        return [self._word_to_index.get(token, self._UNKNOWN_TOKEN_ID) for token in tokens]

    def decode(self, ids: list[int], sep: str = " ") -> str:
        """Decodes a list of integer IDs back into a string using the vocabulary.

        Args:
            ids (list[int]): A list of integer IDs to be decoded.
            sep (str, optional): The separator used to join the decoded tokens. Defaults to a single space.

        Returns:
            str: The decoded string from the list of integer IDs.

        Raises:
            ValueError: If an ID is not found in the reverse vocabulary.
        """
        try:
            words = [self._index_to_word[identifier] for identifier in ids]
        except KeyError as err:
            raise ValueError(f"Identifier '{err.args[0]}' not found in reverse vocabulary.") from err

        text = sep.join(words)
        # Remove spaces before punctuation
        return re.sub(r"\s+([-.,;:?!'\"\)\]])", r"\1", text)
