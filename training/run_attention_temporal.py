from __future__ import annotations

import numpy as np

from training.train_attention_temporal import train_one_run


def main():
    windows = np.load("data/processed/windows.npy", allow_pickle=True)
    labels = np.load("data/processed/labels.npy", allow_pickle=True)
    subjects = np.load("data/processed/subjects.npy", allow_pickle=True)

    print("windows shape :", windows.shape)
    print("labels shape  :", labels.shape)
    print("subjects shape:", subjects.shape)
    print("unique subjects:", np.unique(subjects))
    print("unique labels:", np.unique(labels))

    # Hold out one subject for testing
    test_subject = "S2"

    train_mask = subjects != test_subject
    test_mask = subjects == test_subject

    train_windows = windows[train_mask]
    train_labels = labels[train_mask]
    train_subjects = subjects[train_mask]

    test_windows = windows[test_mask]
    test_labels = labels[test_mask]
    test_subjects = subjects[test_mask]

    print(f"\nUsing subject {test_subject} as test subject")
    print("train_windows:", train_windows.shape)
    print("test_windows :", test_windows.shape)

    train_one_run(
        train_windows=train_windows,
        train_labels=train_labels,
        train_subjects=train_subjects,
        test_windows=test_windows,
        test_labels=test_labels,
        test_subjects=test_subjects,
        sequence_length=4,
        batch_size=64,
        epochs=25,
        lr=1e-3,
        latent_dim=64,
        gru_hidden_dim=128,
        checkpoint_dir="checkpoints",
    )


if __name__ == "__main__":
    main()