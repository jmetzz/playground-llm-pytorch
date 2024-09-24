import pytest

from tokenizers import SimpleRegexTokenizerV1


def test_simple_regex_tokenizer_encode():
    vocabulary = {"Hello": 1, "world": 2, "!": 3}
    tokenizer = SimpleRegexTokenizerV1(vocabulary)
    text = "Hello world!"
    result = tokenizer.encode(text)
    assert result == [1, 2, 3]


def test_simple_regex_tokenizer_encode_key_error():
    vocabulary = {"Hello": 1, "world": 2}
    tokenizer = SimpleRegexTokenizerV1(vocabulary)
    text = "Hello world!"
    with pytest.raises(ValueError, match="Token '!' not found in vocabulary."):
        tokenizer.encode(text)


def test_simple_regex_tokenizer_decode():
    vocabulary = {"Hello": 1, "world": 2, "!": 3}
    tokenizer = SimpleRegexTokenizerV1(vocabulary)
    result = tokenizer.decode([1, 2, 3])
    assert result == "Hello world!"


def test_simple_regex_tokenizer_decode_key_error():
    vocabulary = {"Hello": 1, "world": 2}
    tokenizer = SimpleRegexTokenizerV1(vocabulary)
    with pytest.raises(ValueError, match="Identifier '3' not found in reverse vocabulary."):
        tokenizer.decode([1, 2, 3])
