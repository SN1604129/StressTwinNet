from __future__ import annotations

from pathlib import Path
import numpy as np

from utils.data_loader import load_wesad_all
from utils.preprocess import preprocess_subject_signals, PreprocessConfig
from utils.windowing import make_windows, WindowConfig


RAW_DATA_DIR = "data/raw/WESAD"
OUT_DIR = Path("data/processed")

# Final multimodal order used everywhere in the project
CHANNEL_ORDER = ["ECG", "EDA", "RESP", "BVP"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    preprocess_cfg = PreprocessConfig(target_rate_hz=32)
    window_cfg = WindowConfig(
        target_rate_hz=32,
        window_sec=30,
        stride_sec=5,
        keep_labels=(1, 2),   # baseline, stress
        min_label_fraction=0.6,
    )

    all_windows = []
    all_labels = []
    all_subjects = []

    subjects = load_wesad_all(RAW_DATA_DIR)
    print(f"Loaded {len(subjects)} subjects")

    for subject in subjects:
        print(f"Processing subject {subject.subject_id}...")

        processed = preprocess_subject_signals(
            subject.signals,
            subject.sampling_rates_hz,
            preprocess_cfg,
        )

        windows, labels = make_windows(
            processed,
            subject.labels,
            window_cfg,
            CHANNEL_ORDER,
        )

        subject_ids = np.full((len(labels),), subject.subject_id, dtype=object)

        all_windows.append(windows.astype(np.float32))
        all_labels.append(labels.astype(np.int64))
        all_subjects.append(subject_ids)

        unique, counts = np.unique(labels, return_counts=True) if len(labels) > 0 else ([], [])
        label_info = dict(zip(unique.tolist(), counts.tolist())) if len(labels) > 0 else {}
        print(f"  windows={windows.shape}, labels={labels.shape}, label_counts={label_info}")

    if len(all_windows) == 0:
        raise RuntimeError("No windows were created. Check preprocessing, labels, and channel availability.")

    windows = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    subjects_arr = np.concatenate(all_subjects, axis=0)

    print("\nFinal shapes:")
    print("windows :", windows.shape)
    print("labels  :", labels.shape)
    print("subjects:", subjects_arr.shape)

    np.save(OUT_DIR / "windows.npy", windows)
    np.save(OUT_DIR / "labels.npy", labels)
    np.save(OUT_DIR / "subjects.npy", subjects_arr)

    print("\nSaved:")
    print(OUT_DIR / "windows.npy")
    print(OUT_DIR / "labels.npy")
    print(OUT_DIR / "subjects.npy")


if __name__ == "__main__":
    main()