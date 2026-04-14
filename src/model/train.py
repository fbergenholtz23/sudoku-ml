"""Training loop."""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from src.model.network import SudokuNet
from src.model.dataset import SudokuStepDataset


def train(
    boards: "np.ndarray",
    rows: "np.ndarray",
    cols: "np.ndarray",
    digits: "np.ndarray",
    strategies: "np.ndarray",
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    device: str | None = None,
    checkpoint_path: str = "checkpoints/model.pt",
) -> SudokuNet:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Training on {device}")
    pin_memory = device == "cuda"

    # Use multiple DataLoader workers on Linux/CUDA; stay at 0 on macOS
    # (macOS multiprocessing with spawn causes DataLoader hangs)
    if device == "cuda":
        # 4090 is a beast; use more workers and a much larger batch
        num_workers = 12
        effective_batch = 8192
    else:
        num_workers = 0
        effective_batch = batch_size

    dataset = SudokuStepDataset(boards, rows, cols, digits)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Vectorised weight computation (avoid slow Python loop over millions of items)
    train_strategies = strategies[:train_size]
    unique, counts = np.unique(train_strategies, return_counts=True)
    freq = dict(zip(unique, counts))
    
    # Use NumPy mapping instead of a list comprehension for speed
    weight_map = np.array([1.0 / freq[s] for s in unique])
    strat_to_idx_map = {s: i for i, s in enumerate(unique)}
    # Convert strategy names to indices first, then map to weights
    strat_indices = np.vectorize(strat_to_idx_map.get)(train_strategies)
    sample_weights = torch.from_numpy(weight_map[strat_indices]).float()
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=train_size, replacement=True)
    if len(unique) > 1:
        print(f"Oversampling strategies (total={train_size}): " + ", ".join(f"{s}={c}" for s, c in zip(unique, counts)))

    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, batch_size=effective_batch, sampler=sampler, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=effective_batch, **loader_kwargs)

    model = SudokuNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    from tqdm import tqdm

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x).view(-1, 81 * 9)  # (batch, 729)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                logits = model(x).view(-1, 81 * 9)
                val_loss += criterion(logits, y).item() * len(x)
                correct += (logits.argmax(1) == y).sum().item()

        train_loss /= train_size
        val_loss /= val_size
        acc = correct / val_size
        scheduler.step()

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            import os
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    return model
