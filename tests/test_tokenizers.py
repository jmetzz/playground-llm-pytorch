import pytest

from tokenizers import SimpleRegexTokenizerV1


def test_simple_regex_tokenizer_encode():
    tokenizer = SimpleRegexTokenizerV1(vocabulary=["Hello", "world", "!"])
    actual = tokenizer.encode("Hello world!")
    assert actual == [1, 2, 0]


def test_simple_regex_tokenizer_decode():
    tokenizer = SimpleRegexTokenizerV1(vocabulary=["Hello", "world", "!"])
    actual = tokenizer.decode([1, 2, 0])
    assert actual == "Hello world!"


def test_simple_regex_tokenizer_decode_key_error():
    tokenizer = SimpleRegexTokenizerV1(vocabulary=["Hello", "world"])
    with pytest.raises(ValueError, match="Identifier '2' not found in reverse vocabulary."):
        tokenizer.decode([0, 1, 2])


def test_simple_regex_tokenizer_encode_key_error():
    tokenizer = SimpleRegexTokenizerV1(vocabulary=["Hello", "world"])
    text = "Hello world!"
    with pytest.raises(ValueError, match="Token '!' not found in vocabulary."):
        tokenizer.encode(text)
