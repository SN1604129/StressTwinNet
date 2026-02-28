from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

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
    channel_order: List[str] = field(default_factory=lambda: ["ECG", "EDA", "RESP", "BVP"])


def build_subject_windows(
    subject: SubjectData,
    cfg: Optional[DatasetConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build windows for a single subject (preprocess -> windowing).
    Returns:
      X: float32 [N, C, T]
      y: int64   [N] raw labels (expected 1 or 2 only if your windowing filters correctly)
    """
    cfg = cfg or DatasetConfig()

    processed = preprocess_subject_signals(
        subject.signals,
        subject.sampling_rates_hz,
        cfg.preprocess,
    )

    X, y = make_windows(
        processed,
        subject.labels,
        cfg.window,
        cfg.channel_order,
    )

    if X.size > 0:
        X = X.astype(np.float32, copy=False)
    if y.size > 0:
        y = y.astype(np.int64, copy=False)

    return X, y


def build_subject_splits(
    subjects: Sequence[SubjectData],
    train_subject_ids: Sequence[str],
    test_subject_ids: Sequence[str],
    cfg: Optional[DatasetConfig] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Subject-independent split builder.

    IMPORTANT:
      We split by subject first, THEN build windows.
      This prevents leakage where overlapping windows from the same subject appear in both train and test.

    Returns:
      (X_train, y_train), (X_test, y_test)
    """
    cfg = cfg or DatasetConfig()

    subj_map = {s.subject_id: s for s in subjects}

    missing_train = [sid for sid in train_subject_ids if sid not in subj_map]
    missing_test = [sid for sid in test_subject_ids if sid not in subj_map]
    if missing_train or missing_test:
        raise ValueError(
            f"Missing subjects. "
            f"Missing train={missing_train}, missing test={missing_test}. "
            f"Available={sorted(subj_map.keys())}"
        )

    Xtr_list, ytr_list = [], []
    Xte_list, yte_list = [], []

    for sid in train_subject_ids:
        X, y = build_subject_windows(subj_map[sid], cfg)
        if X.shape[0] > 0:
            Xtr_list.append(X)
            ytr_list.append(y)

    for sid in test_subject_ids:
        X, y = build_subject_windows(subj_map[sid], cfg)
        if X.shape[0] > 0:
            Xte_list.append(X)
            yte_list.append(y)

    def _cat_or_empty(X_list: List[np.ndarray], y_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if len(X_list) > 0:
            Xc = np.concatenate(X_list, axis=0).astype(np.float32, copy=False)
            yc = np.concatenate(y_list, axis=0).astype(np.int64, copy=False)
            return Xc, yc

        win_len = cfg.window.window_sec * cfg.window.target_rate_hz
        X_empty = np.zeros((0, len(cfg.channel_order), win_len), dtype=np.float32)
        y_empty = np.zeros((0,), dtype=np.int64)
        return X_empty, y_empty

    X_train, y_train = _cat_or_empty(Xtr_list, ytr_list)
    X_test, y_test = _cat_or_empty(Xte_list, yte_list)

    return (X_train, y_train), (X_test, y_test)


class StressWindowDataset(Dataset):
    """
    PyTorch Dataset that:
      1) preprocesses raw signals
      2) creates sliding windows
      3) returns (C, T) tensors with integer labels

    Label mapping (binary):
      WESAD baseline=1 -> 0
      WESAD stress=2   -> 1

    Output:
      x: torch.FloatTensor (C, T)
      y: torch.LongTensor scalar (0 or 1)
    """

    def __init__(self, subjects: List[SubjectData], cfg: Optional[DatasetConfig] = None):
        self.cfg = cfg or DatasetConfig()

        X_all: List[np.ndarray] = []
        y_all: List[np.ndarray] = []

        for subject in subjects:
            X, y = build_subject_windows(subject, self.cfg)
            if X.shape[0] > 0:
                X_all.append(X)
                y_all.append(y)

        if len(X_all) > 0:
            self.X = np.concatenate(X_all, axis=0).astype(np.float32, copy=False)
            self.y = np.concatenate(y_all, axis=0).astype(np.int64, copy=False)
        else:
            win_len = self.cfg.window.window_sec * self.cfg.window.target_rate_hz
            self.X = np.zeros((0, len(self.cfg.channel_order), win_len), dtype=np.float32)
            self.y = np.zeros((0,), dtype=np.int64)

    @classmethod
    def from_arrays(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        cfg: Optional[DatasetConfig] = None,
    ) -> "StressWindowDataset":
        """
        Build dataset from precomputed arrays (useful for subject split outputs).
        X: [N, C, T] float32
        y: [N] int64 (raw labels 1 or 2)
        """
        obj = cls.__new__(cls)
        obj.cfg = cfg or DatasetConfig()
        obj.X = np.asarray(X, dtype=np.float32)
        obj.y = np.asarray(y, dtype=np.int64)

        if obj.X.ndim != 3:
            raise ValueError(f"X must be 3D [N,C,T]. Got {obj.X.shape}")
        if obj.y.ndim != 1 or obj.y.shape[0] != obj.X.shape[0]:
            raise ValueError(f"y must be 1D with same N as X. Got X={obj.X.shape}, y={obj.y.shape}")

        return obj

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).float()

        y_raw = int(self.y[idx])
        if y_raw == 1:
            y_bin = 0  # baseline
        elif y_raw == 2:
            y_bin = 1  # stress
        else:
            raise ValueError(
                f"Unexpected label {y_raw}. "
                "Windowing should keep only labels (1=baseline, 2=stress)."
            )

        y = torch.tensor(y_bin, dtype=torch.long)
        return x, y