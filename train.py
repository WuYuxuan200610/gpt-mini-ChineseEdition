"""Train a mini GPT on a JSONL corpus.

Usage:
    python train.py \
        --data_path data/mobvoi_seq_monkey_general_open_corpus.jsonl \
        --block_size 512 --batch_size 12 --max_lines 1000 \
        --n_layer 6 --n_head 6 --n_embd 384 \
        --epochs 2 --lr 3e-4 --out_dir checkpoints
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader, random_split

from config import GPTConfig
from dataset import MyDataset
from model import GPT
from utils import set_seed, get_device, count_parameters


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    log_every: int = 100,
) -> float:
    """Run one training epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if step % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"[epoch {epoch}] step {step:4d}/{len(loader)} "
                f"loss={loss.item():.4f} lr={lr:.2e}"
            )
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return the average validation loss."""
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def build_lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup followed by cosine decay."""
    def _fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return _fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a mini GPT.")
    # data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_lines", type=int, default=1000)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    # model
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    # training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # io
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    # ---- dataset -------------------------------------------------------
    dataset = MyDataset(
        args.data_path,
        block_size=args.block_size,
        max_lines=args.max_lines,
    )
    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Chunks: total={len(dataset)}, train={n_train}, val={n_val}")

    # ---- model ---------------------------------------------------------
    config = GPTConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    print(f"Parameters: {count_parameters(model) / 1e6:.2f}M")

    # ---- optimizer & scheduler ----------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=build_lr_lambda(args.warmup_steps, total_steps)
    )

    # ---- training loop -------------------------------------------------
    best_val = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch,
            grad_clip=args.grad_clip,
        )
        val_loss = evaluate(model, val_loader, device)
        print(
            f"== epoch {epoch} | train={train_loss:.4f} "
            f"| val={val_loss:.4f} | {time.time() - t0:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss,
            "config": asdict(config),
        }
        ckpt_path = os.path.join(args.out_dir, f"model_epoch_{epoch}.pt")
        torch.save(checkpoint, ckpt_path)
        print(f"Saved {ckpt_path}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, os.path.join(args.out_dir, "best.pt"))


if __name__ == "__main__":
    main()