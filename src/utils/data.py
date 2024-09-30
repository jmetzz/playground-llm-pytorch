from pathlib import Path
from typing import Any

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


def load_text_file(file_path: Path) -> str:
    with file_path.open(mode="r", encoding="utf-8") as f:
        return f.read()


def get_encoder_and_batch_iterator(
    file_path: Path,
    encoding: str = "gpt2",
    batch_size: int = 8,
    stride: int = 1,
    seq_length: int = 1,
    shuffle: bool = False,
) -> tuple:
    content = load_text_file(file_path)
    encoder = tiktoken.get_encoding(encoding_name=encoding)
    data_loader = create_dataloader_v1(
        text=content, encoder=encoder, batch_size=batch_size, stride=stride, seq_length=seq_length, shuffle=shuffle
    )
    return encoder, data_loader


class GPTDatasetV1(Dataset):
    def __init__(self, text: str, encoder: Any, max_length: int, stride: int) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = encoder.encode(text)
        for idx in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[idx : idx + max_length]
            target_chunk = token_ids[idx + 1 : idx + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(  # noqa: PLR0913, PLR0917
    text: str,
    encoder: Any,
    batch_size: int = 4,
    seq_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> tuple[Dataset, DataLoader]:
    dataset = GPTDatasetV1(text, encoder, seq_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
