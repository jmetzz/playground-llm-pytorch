import torch


def tokenize(  # noqa: PLR0913
    sentence: str,
    language_index: dict[str, int],
    max_seq_length: int = 200,
    *,
    start_token: str | None = None,
    end_token: str | None = None,
    padding_token: str | None = None,
) -> torch.Tensor:
    start_token = start_token or ""
    end_token = end_token or ""
    sentence_index = [language_index[token] for token in sentence.split(" ") if token.strip()]
    if start_token:
        sentence_index.insert(0, language_index[start_token])
    if end_token:
        sentence_index.append(language_index[end_token])

    # Ensure the tokenized sentence fits the allowed sequence length
    # by truncate and padding if necessary
    sentence_index = sentence_index[:max_seq_length]
    padding_size = max_seq_length - len(sentence_index)
    padding_arr = [language_index[padding_token]] * padding_size
    return torch.tensor(sentence_index + padding_arr, dtype=torch.int)


def tokenize_batch(  # noqa: PLR0913
    batch: list[str],
    language_index: dict[str, int],
    max_seq_length: int = 200,
    *,
    start_token: str | None = None,
    end_token: str | None = None,
    padding_token: str | None = None,
) -> torch.Tensor:
    tokenized = [
        tokenize(
            sentence=batch[sentence_num],
            language_index=language_index,
            max_seq_length=max_seq_length,
            start_token=start_token,
            end_token=end_token,
            padding_token=padding_token,
        )
        for sentence_num in range(len(batch))
    ]

    return torch.stack(tokenized)
