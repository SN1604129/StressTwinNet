from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceSample:
    x: np.ndarray   # [S, M, T]
    y: int


class StressSequenceDataset(Dataset):
    """
    Converts window-level samples into sequence-level samples.

    Expected inputs:
      windows: np.ndarray [N, M, T]
      labels:  np.ndarray [N]   # WESAD labels: 1=baseline, 2=stress
      subjects: np.ndarray [N]
      sequence_length: number of consecutive windows per sample

    Only sequences fully within the same subject are kept.
    Final label = label of last timestep.

    IMPORTANT:
      Maps WESAD labels:
        1 -> 0
        2 -> 1
    """
    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        subjects: np.ndarray,
        sequence_length: int = 4,
    ):
        self.samples: List[SequenceSample] = []

        n = len(windows)
        for i in range(sequence_length - 1, n):
            start = i - sequence_length + 1
            end = i + 1

            seq_subjects = subjects[start:end]
            if not np.all(seq_subjects == seq_subjects[0]):
                continue

            x_seq = windows[start:end]   # [S, M, T]
            y_raw = int(labels[i])

            # Map WESAD labels to 0-based class indices for CrossEntropyLoss
            if y_raw == 1:
                y_seq = 0   # baseline
            elif y_raw == 2:
                y_seq = 1   # stress
            else:
                continue

            self.samples.append(
                SequenceSample(x=x_seq.astype(np.float32), y=y_seq)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        x = torch.tensor(sample.x, dtype=torch.float32)   # [S, M, T]
        y = torch.tensor(sample.y, dtype=torch.long)
        return x, y