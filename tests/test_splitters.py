from splitters import punctuation_splitter, space_splitter


def test_space_splitter_encode_space():
    text = "Hello world!"
    result = space_splitter(text, encode_spaces=True)
    assert result == ["Hello", " ", "world!"]


def test_space_splitter_no_encode_space():
    text = "Hello world!"
    result = space_splitter(text, encode_spaces=False)
    assert result == ["Hello", "world!"]


def test_punctuation_splitter_encode_space():
    text = "Hello, world!"
    result = punctuation_splitter(text, encode_spaces=True)
    assert result == ["Hello", ",", " world", "!"]


def test_punctuation_splitter_no_encode_space():
    text = "Hello, world!"
    result = punctuation_splitter(text, encode_spaces=False)
    assert result == ["Hello", ",", " world", "!"]
