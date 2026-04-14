"""Training loop."""

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
    pin_memory = device == "cuda"  # MPS and CPU don't support pin_memory

    import numpy as np
    dataset = SudokuStepDataset(boards, rows, cols, digits)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Oversample rare strategies so the model sees them proportionally
    train_strategies = strategies[:train_size]
    unique, counts = np.unique(train_strategies, return_counts=True)
    freq = dict(zip(unique, counts))
    sample_weights = torch.tensor(
        [1.0 / freq[train_strategies[i]] for i in range(train_size)],
        dtype=torch.float,
    )
    sampler = WeightedRandomSampler(sample_weights, num_samples=train_size, replacement=True)
    if len(unique) > 1:
        print("Oversampling strategies: " + ", ".join(f"{s}={c}" for s, c in zip(unique, counts)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    model = SudokuNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).view(-1, 81 * 9)  # (batch, 729)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
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
