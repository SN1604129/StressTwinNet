import os
import glob
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# Common WESAD sampling rates (typical WESAD release)
# Chest signals are usually high-rate (e.g., 700 Hz). Wrist signals are lower-rate.
DEFAULT_SAMPLING_RATES_HZ = {
    "chest": {
        "ECG": 700,
        "EDA": 700,
        "RESP": 700,
    },
    "wrist": {
        "BVP": 64,   # PPG/BVP
        "EDA": 4,    # if present
    },
}

# WESAD labels (typical):
# 0=not defined, 1=baseline, 2=stress, 3=amusement, 4=meditation
# For this project we focus on baseline/stress/recovery-like phases later.
LABEL_MAP = {
    0: "undefined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}


@dataclass
class SubjectData:
    """Container for one subject's raw signals + labels."""
    subject_id: str
    signals: Dict[str, Dict[str, np.ndarray]]  # modality -> channel -> 1D array
    labels: np.ndarray                         # 1D int array
    sampling_rates_hz: Dict[str, Dict[str, int]]


def _load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _as_1d(x) -> np.ndarray:
    """
    Ensure signal is a 1D float32 numpy array.
    WESAD sometimes stores arrays as (N,1) or (N,) depending on channel.
    """
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def find_subject_pkls(wesad_root: str, subjects: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Find subject pickle files under:
      data/raw/WESAD/S2/S2.pkl, ... etc.

    Returns list of (subject_id, pkl_path).
    """
    pattern = os.path.join(wesad_root, "S*", "S*.pkl")
    paths = sorted(glob.glob(pattern))

    found = []
    for p in paths:
        sid = os.path.basename(os.path.dirname(p))  # S2
        if subjects is not None and sid not in subjects:
            continue
        found.append((sid, p))

    if not found:
        raise FileNotFoundError(
            f"No subject .pkl files found under {wesad_root}. "
            f"Expected structure like {wesad_root}/S2/S2.pkl"
        )

    return found


def load_wesad_subject(pkl_path: str, subject_id: Optional[str] = None) -> SubjectData:
    """
    Load a single WESAD subject pickle.

    Expected WESAD structure (common):
      data['signal']['chest']['ECG'] -> array
      data['signal']['chest']['EDA'] -> array
      data['signal']['chest']['Resp'] or 'RESP' -> array (naming varies)
      data['signal']['wrist']['BVP'] -> array
      data['signal']['wrist']['EDA'] -> array (optional)
      data['label'] -> array of int labels

    We normalize key variations safely.
    """
    raw = _load_pickle(pkl_path)

    if subject_id is None:
        subject_id = os.path.basename(os.path.dirname(pkl_path))

    if "signal" not in raw or "label" not in raw:
        raise KeyError(f"{pkl_path} does not look like a WESAD subject file (missing 'signal' or 'label').")

    signal = raw["signal"]
    labels = np.asarray(raw["label"]).astype(np.int64)

    # Extract desired channels safely
    signals: Dict[str, Dict[str, np.ndarray]] = {"chest": {}, "wrist": {}}

    # ---- Chest ----
    chest = signal.get("chest", {})
    for key in ["ECG", "EDA"]:
        if key in chest:
            signals["chest"][key] = _as_1d(chest[key])

    # Resp can be stored as "Resp" in some dumps
    if "RESP" in chest:
        signals["chest"]["RESP"] = _as_1d(chest["RESP"])
    elif "Resp" in chest:
        signals["chest"]["RESP"] = _as_1d(chest["Resp"])

    # ---- Wrist ----
    wrist = signal.get("wrist", {})
    if "BVP" in wrist:
        signals["wrist"]["BVP"] = _as_1d(wrist["BVP"])
    # optional
    if "EDA" in wrist:
        signals["wrist"]["EDA"] = _as_1d(wrist["EDA"])

    # Basic checks
    if len(signals["chest"]) == 0 and len(signals["wrist"]) == 0:
        raise ValueError(f"No expected channels found in {pkl_path}.")

    return SubjectData(
        subject_id=subject_id,
        signals=signals,
        labels=labels,
        sampling_rates_hz=DEFAULT_SAMPLING_RATES_HZ,
    )


def load_wesad_all(
    wesad_root: str,
    subjects: Optional[List[str]] = None,
) -> List[SubjectData]:
    """
    Load multiple subjects (e.g., S2–S17). If subjects=None, loads all found.
    """
    items = find_subject_pkls(wesad_root, subjects=subjects)
    out = []
    for sid, p in items:
        out.append(load_wesad_subject(p, subject_id=sid))
    return out


def to_torch(subject: SubjectData) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Convert numpy signals to torch tensors (float32).
    Labels remain numpy in SubjectData; you can convert separately if needed.
    """
    t: Dict[str, Dict[str, torch.Tensor]] = {"chest": {}, "wrist": {}}
    for mod, chans in subject.signals.items():
        for ch, arr in chans.items():
            t[mod][ch] = torch.from_numpy(arr.astype(np.float32, copy=False))
    return t


if __name__ == "__main__":
    # Sanity test: load one subject (S2 if available)
    root = os.environ.get("WESAD_ROOT", "data/raw/WESAD")
    subjects = None

    subs = load_wesad_all(root, subjects=subjects)
    s0 = subs[0]
    print("Loaded subject:", s0.subject_id)
    print("Channels:")
    for mod in ["chest", "wrist"]:
        for ch, arr in s0.signals[mod].items():
            print(f"  {mod}/{ch}: shape={arr.shape}, dtype={arr.dtype}, sr={s0.sampling_rates_hz[mod].get(ch)}")
    print("Labels:", s0.labels.shape, "unique:", np.unique(s0.labels))