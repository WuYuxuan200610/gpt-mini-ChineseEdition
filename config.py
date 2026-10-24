"""Model configuration for the mini GPT implementation.

This module only defines *model architecture* hyper-parameters.
Training-related hyper-parameters (epochs, lr, data path, ...) live in
``train.py`` / ``argparse`` so that the same config can be reused for both
training and inference.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Hyper-parameters that define the shape of the GPT model.

    Attributes:
        block_size: Maximum context length (number of tokens).
        n_layer: Number of stacked Transformer blocks.
        n_head: Number of attention heads in each block.
        n_embd: Hidden dimension of the residual stream.
        dropout: Dropout probability used in attention and MLP.
        vocab_size: Tokenizer vocabulary size (GPT-2 BPE -> 50257).
    """

    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    vocab_size: int = 50257

    @property
    def head_size(self) -> int:
        """Dimension of each attention head (``n_embd // n_head``)."""
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by "
                f"n_head ({self.n_head})"
            )
        return self.n_embd // self.n_head

