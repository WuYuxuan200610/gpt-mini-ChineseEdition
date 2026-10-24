"""Language-modeling dataset.

Reads a JSONL file (one ``{"text": ...}`` per line), tokenizes with the
GPT-2 BPE tokenizer, concatenates documents with ``<|endoftext|>``, and
splits the long token stream into fixed-length chunks.

Each item is ``(x, y)`` with shape ``(block_size,)``; ``y`` is ``x``
shifted by one token.
"""

from __future__ import annotations

import json
from typing import List

import torch
import tiktoken
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """Fixed-length next-token-prediction dataset."""

    def __init__(self, path: str, block_size: int = 512, max_lines: int = 1000):
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.max_lines = max_lines

        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"},
        )[0]

        # Read up to max_lines valid JSONL records
        raw_data: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    raw_data.append(json.loads(line.strip())["text"])
                except json.JSONDecodeError:
                    continue

        # Flatten all documents into one long token stream
        full_encoded: List[int] = []
        for text in raw_data:
            full_encoded.extend(self.enc.encode(text))
            full_encoded.append(self.eos_token)

        # Split into chunks of (block_size + 1) tokens.
        # Drop the trailing partial chunk so that all items share the same
        # length (required by ``default_collate`` in DataLoader).
        self.encoded_data: List[List[int]] = []
        for i in range(0, len(full_encoded) - self.block_size, self.block_size):
            chunk = full_encoded[i:i + self.block_size + 1]
            self.encoded_data.append(chunk)

        assert len(self.encoded_data) > 0, (
            f"No valid chunks produced (len={len(full_encoded)}, "
            f"block_size={self.block_size}). Check data path / max_lines."
        )

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, idx: int):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)