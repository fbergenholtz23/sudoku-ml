"""CNN model for sudoku move prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


def encode_boards(boards: torch.Tensor) -> torch.Tensor:
    """
    Vectorized board encoding (placed digits + candidates) on GPU.
    Input:  (batch, 9, 9) int64/long tensor, values 0-9
    Output: (batch, 18, 9, 9) float32 tensor
    """
    batch_size = boards.shape[0]
    # Channels 0-8: one-hot for digits 1-9
    # boards is 0-9; we want one-hot for 1-9.
    placed = F.one_hot(boards, num_classes=10)[..., 1:]  # (B, 9, 9, 9)
    placed = placed.permute(0, 3, 1, 2).float()  # (B, 9, 9, 9)

    # Channels 9-17: candidates
    # A digit d is a candidate if it's not in the same row, col, or box
    empty = (boards == 0).unsqueeze(1).float()  # (B, 1, 9, 9)

    # row_conflict: (B, 9, 9, 1) -> (B, 9, 9, 9)
    row_conflict = placed.any(dim=3, keepdim=True)
    # col_conflict: (B, 9, 1, 9) -> (B, 9, 9, 9)
    col_conflict = placed.any(dim=2, keepdim=True)

    # box_conflict: (B, 9, 3, 3) -> (B, 9, 9, 9)
    boxes = placed.view(batch_size, 9, 3, 3, 3, 3)
    box_conflict = boxes.any(dim=(4, 5), keepdim=True).expand(-1, -1, -1, -1, 3, 3).reshape(batch_size, 9, 9, 9)

    candidates = empty * (1.0 - (row_conflict | col_conflict | box_conflict.bool()).float())

    return torch.cat([placed, candidates], dim=1)


class SudokuNet(nn.Module):
    """
    Input:  (batch, 18, 9, 9) or (batch, 9, 9)
              - channels 0–8:  one-hot placed digits
              - channels 9–17: candidate digits per cell

    Output: (batch, 81, 9)  — logits over digits for each cell.
            During inference, mask out already-filled cells and take argmax.
    """

    def __init__(self, channels: int = 128, num_res_blocks: int = 12):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(18, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])
        
        # Version 1 (Small) used 32 channels and no BatchNorm in the head
        # Version 2 (Large) uses 64 channels and BatchNorm
        head_channels = 64 if channels > 64 else 32
        use_bn = channels > 64
        
        layers = [nn.Conv2d(channels, head_channels, kernel_size=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(head_channels))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(head_channels * 9 * 9, 81 * 9),
        ])
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:  # (batch, 9, 9) raw boards
            x = encode_boards(x)
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.head(x).view(-1, 81, 9)
