"""Generate text from a trained checkpoint.

Usage:
    python generate.py --ckpt checkpoints/best.pt \
        --prompt "今天天气" --max_new_tokens 200
"""

from __future__ import annotations

import argparse

import torch
import tiktoken

from config import GPTConfig
from model import GPT
from utils import get_device




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    if args.prompt:
        idx = torch.tensor(
            [enc.encode(args.prompt)], dtype=torch.long, device=device
        )
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    out = model.generate(idx, max_new_tokens=args.max_new_tokens)
    print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()