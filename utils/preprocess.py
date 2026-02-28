from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from scipy import signal


@dataclass
class PreprocessConfig:
    """
    Preprocessing configuration.

    target_rate_hz:
      We align all channels to this rate. For early development, 32 Hz is a good
      compromise (keeps wrist ACC-like rates compatible and dramatically reduces chest size).
    """
    target_rate_hz: int = 32

    # Filtering options (basic, reviewer-safe)
    # ECG: bandpass 0.5–40 Hz
    ecg_bandpass_hz: Tuple[float, float] = (0.5, 40.0)

    # EDA: lowpass ~5 Hz (EDA is slow)
    eda_lowpass_hz: float = 5.0

    # RESP: bandpass ~0.1–0.7 Hz (breathing is slow)
    resp_bandpass_hz: Tuple[float, float] = (0.1, 0.7)

    # Normalization
    per_subject_zscore: bool = True


def _resample_1d(x: np.ndarray, orig_hz: int, target_hz: int) -> np.ndarray:
    """
    Resample a 1D signal from orig_hz to target_hz using polyphase filtering
    (more stable than naive interpolation for large ratios).
    """
    if orig_hz == target_hz:
        return x.astype(np.float32, copy=False)

    # Reduce ratio
    g = np.gcd(orig_hz, target_hz)
    up = target_hz // g
    down = orig_hz // g

    y = signal.resample_poly(x, up=up, down=down).astype(np.float32, copy=False)
    return y


def _butter_filter(
    x: np.ndarray,
    fs: int,
    kind: str,
    cutoff: Tuple[float, float] | float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply Butterworth filter (zero-phase) for basic physiological cleanup.
    """
    nyq = 0.5 * fs

    if kind == "bandpass":
        low, high = cutoff  # type: ignore
        low = max(low / nyq, 1e-6)
        high = min(high / nyq, 0.999999)
        b, a = signal.butter(order, [low, high], btype="bandpass")
    elif kind == "lowpass":
        c = cutoff  # type: ignore
        c = min(c / nyq, 0.999999)
        b, a = signal.butter(order, c, btype="lowpass")
    else:
        raise ValueError(f"Unknown filter kind: {kind}")

    return signal.filtfilt(b, a, x).astype(np.float32, copy=False)


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return ((x - mu) / (sd + eps)).astype(np.float32, copy=False)


def preprocess_subject_signals(
    signals: Dict[str, Dict[str, np.ndarray]],
    sampling_rates_hz: Dict[str, Dict[str, int]],
    cfg: Optional[PreprocessConfig] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Preprocess raw WESAD signals:
      1) resample to cfg.target_rate_hz for alignment across modalities
      2) basic filtering per channel type
      3) z-score normalization (per subject)

    Returns same structure: modality -> channel -> 1D float32 array.
    """
    if cfg is None:
        cfg = PreprocessConfig()

    out: Dict[str, Dict[str, np.ndarray]] = {"chest": {}, "wrist": {}}
    target = cfg.target_rate_hz

    # -------- Chest --------
    for ch, x in signals.get("chest", {}).items():
        orig = sampling_rates_hz["chest"].get(ch)
        if orig is None:
            raise KeyError(f"Missing sampling rate for chest/{ch}")

        # Resample first (reduces compute)
        y = _resample_1d(x, orig_hz=orig, target_hz=target)

        # Filter
        if ch == "ECG":
            y = _butter_filter(y, fs=target, kind="bandpass", cutoff=cfg.ecg_bandpass_hz)
        elif ch == "EDA":
            y = _butter_filter(y, fs=target, kind="lowpass", cutoff=cfg.eda_lowpass_hz)
        elif ch == "RESP":
            y = _butter_filter(y, fs=target, kind="bandpass", cutoff=cfg.resp_bandpass_hz)

        out["chest"][ch] = y

    # -------- Wrist --------
    for ch, x in signals.get("wrist", {}).items():
        orig = sampling_rates_hz["wrist"].get(ch)
        if orig is None:
            raise KeyError(f"Missing sampling rate for wrist/{ch}")

        y = _resample_1d(x, orig_hz=orig, target_hz=target)

        # Basic filters (optional)
        if ch == "BVP":
            # BVP can be noisy; light lowpass for stability
            y = _butter_filter(y, fs=target, kind="lowpass", cutoff=min(8.0, 0.45 * target))
        elif ch == "EDA":
            y = _butter_filter(y, fs=target, kind="lowpass", cutoff=cfg.eda_lowpass_hz)

        out["wrist"][ch] = y

    # -------- Normalize --------
    if cfg.per_subject_zscore:
        for mod in out:
            for ch in out[mod]:
                out[mod][ch] = zscore(out[mod][ch])

    return out


if __name__ == "__main__":
    # Minimal sanity test (requires WESAD signals loaded)
    # This file is primarily called from your pipeline.
    print("preprocess.py loaded OK")