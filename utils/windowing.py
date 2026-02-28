from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class WindowConfig:
    target_rate_hz: int = 32
    window_sec: int = 30
    stride_sec: int = 5
    keep_labels: Tuple[int, ...] = (1, 2)  # baseline=1, stress=2
    min_label_fraction: float = 0.6        # majority threshold


def _stack_channels(channels: Dict[str, np.ndarray], order: List[str]) -> np.ndarray:
    """
    Stack channels into shape (C, T).
    Missing channels raise KeyError to keep things explicit and reviewer-safe.
    """
    xs = []
    for ch in order:
        if ch not in channels:
            raise KeyError(f"Missing channel: {ch}")
        xs.append(channels[ch])
    x = np.stack(xs, axis=0).astype(np.float32, copy=False)
    return x


def _resample_labels_to_length(labels: np.ndarray, target_len: int) -> np.ndarray:
    """
    Map labels from their original time base to a target length using nearest-neighbor index mapping.

    This avoids assuming labels are already at cfg.target_rate_hz.
    WESAD labels often match chest sample length; signals may be resampled to 32Hz.

    Strategy:
      For each target index i in [0, target_len), pick label at round(i * (len(labels)-1)/(target_len-1)).
    """
    labels = np.asarray(labels).astype(np.int64, copy=False)
    n = int(labels.shape[0])

    if target_len <= 0:
        return np.zeros((0,), dtype=np.int64)

    if n == target_len:
        return labels

    if n == 1:
        return np.full((target_len,), int(labels[0]), dtype=np.int64)

    # Create mapping indices from target timeline -> original timeline
    idx = np.linspace(0, n - 1, num=target_len)
    idx = np.rint(idx).astype(np.int64)
    idx = np.clip(idx, 0, n - 1)
    return labels[idx]


def make_windows(
    signals: Dict[str, Dict[str, np.ndarray]],
    labels: np.ndarray,
    cfg: WindowConfig,
    channel_order: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create windows from already-preprocessed signals.

    Inputs:
      signals: modality->channel->1D arrays (ideally at cfg.target_rate_hz)
      labels: 1D int labels (may be at original sampling length)
      channel_order: final channels to include, e.g. ["ECG","EDA","RESP","BVP"]

    Returns:
      X: (N, C, T)
      y: (N,) window labels (int)
    """
    sr = cfg.target_rate_hz
    win = cfg.window_sec * sr
    stride = cfg.stride_sec * sr

    # Flatten modalities into one dict of channels
    flat: Dict[str, np.ndarray] = {}
    for mod in signals:
        for ch, arr in signals[mod].items():
            flat[ch] = arr

    # Ensure all channels exist and compute common length
    lengths = []
    for ch in channel_order:
        if ch not in flat:
            raise KeyError(f"Missing channel: {ch}")
        lengths.append(len(flat[ch]))

    T = int(min(lengths))
    if T < win:
        return (
            np.zeros((0, len(channel_order), win), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    # Trim channels to common length
    for ch in channel_order:
        flat[ch] = flat[ch][:T]

    # Stack ONCE (C, T)
    X_all = _stack_channels(flat, channel_order)  # (C, T)

    # Resample labels to match T (critical fix)
    lbl = _resample_labels_to_length(labels, T)

    X_list = []
    y_list = []

    for start in range(0, T - win + 1, stride):
        end = start + win
        w_labels = lbl[start:end]

        # majority label
        vals, counts = np.unique(w_labels, return_counts=True)
        maj_idx = int(np.argmax(counts))
        maj_label = int(vals[maj_idx])
        frac = float(counts[maj_idx]) / float(win)

        if maj_label in cfg.keep_labels and frac >= cfg.min_label_fraction:
            X_list.append(X_all[:, start:end])
            y_list.append(maj_label)

    if not X_list:
        return (
            np.zeros((0, len(channel_order), win), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    X = np.stack(X_list, axis=0).astype(np.float32, copy=False)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y