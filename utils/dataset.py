from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_loader import SubjectData
from utils.preprocess import preprocess_subject_signals, PreprocessConfig
from utils.windowing import make_windows, WindowConfig


@dataclass
class DatasetConfig:
    """
    Configuration for StressWindowDataset.
    Uses default_factory to avoid mutable default issues (Python 3.12 safe).
    """
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    channel_order: List[str] = field(
        default_factory=lambda: ["ECG", "EDA", "RESP", "BVP"]
    )


class StressWindowDataset(Dataset):
    """
    PyTorch Dataset that:
      1) preprocesses raw signals
      2) creates sliding windows
      3) returns (C, T) tensors with integer labels

    Output:
      x: torch.FloatTensor (C, T)
      y: torch.LongTensor scalar
    """

    def __init__(self, subjects: List[SubjectData], cfg: Optional[DatasetConfig] = None):
        self.cfg = cfg or DatasetConfig()

        X_all = []
        y_all = []

        for subject in subjects:
            # Step 1 — Preprocess signals
            processed = preprocess_subject_signals(
                subject.signals,
                subject.sampling_rates_hz,
                self.cfg.preprocess,
            )

            # Step 2 — Windowing
            X, y = make_windows(
                processed,
                subject.labels,
                self.cfg.window,
                self.cfg.channel_order,
            )

            if X.shape[0] > 0:
                X_all.append(X)
                y_all.append(y)

        if len(X_all) > 0:
            self.X = np.concatenate(X_all, axis=0)
            self.y = np.concatenate(y_all, axis=0)
        else:
            # Empty-safe initialization
            win_len = self.cfg.window.window_sec * self.cfg.window.target_rate_hz
            self.X = np.zeros(
                (0, len(self.cfg.channel_order), win_len),
                dtype=np.float32,
            )
            self.y = np.zeros((0,), dtype=np.int64)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y