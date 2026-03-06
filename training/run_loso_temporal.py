import os
import json
import time
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.data_loader import load_wesad_all
from utils.dataset import DatasetConfig, build_subject_windows
from models.cnn_encoder import EncoderWithHead
from models.temporal_gru import TemporalGRU


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TemporalSequenceDataset(Dataset):
    """
    input  = [z(t-seq_len), ..., z(t-1)]
    target = y(t)
    """
    def __init__(self, Z: np.ndarray, y: np.ndarray, seq_len: int = 4):
        assert Z.ndim == 2
        assert y.ndim == 1
        assert len(Z) == len(y)
        assert len(Z) > seq_len

        self.X_seq = []
        self.y_next = []

        for i in range(seq_len, len(Z)):
            self.X_seq.append(Z[i - seq_len:i])
            self.y_next.append(y[i])

        self.X_seq = np.asarray(self.X_seq, dtype=np.float32)
        self.y_next = np.asarray(self.y_next, dtype=np.int64)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X_seq[idx]).float()
        y = torch.tensor(int(self.y_next[idx]), dtype=torch.long)
        return x, y


class ConcatSeqDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cum = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cum.append(total)

    def __len__(self):
        return self.cum[-1] if self.cum else 0

    def __getitem__(self, idx):
        for i, c in enumerate(self.cum):
            if idx < c:
                prev = 0 if i == 0 else self.cum[i - 1]
                return self.datasets[i][idx - prev]
        raise IndexError


def compute_class_weights_from_binary_y(y_bin: np.ndarray, device: str):
    n0 = int((y_bin == 0).sum())
    n1 = int((y_bin == 1).sum())
    if n0 == 0 or n1 == 0:
        return None
    w0 = (n0 + n1) / (2.0 * n0)
    w1 = (n0 + n1) / (2.0 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def eval_binary_metrics(model, loader, device: str):
    model.eval()
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            tp += int(((preds == 1) & (y == 1)).sum().item())
            tn += int(((preds == 0) & (y == 0)).sum().item())
            fp += int(((preds == 1) & (y == 0)).sum().item())
            fn += int(((preds == 0) & (y == 1)).sum().item())

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    cm = [[tn, fp], [fn, tp]]
    return acc, precision, recall, f1, cm


def load_encoder_checkpoint(ckpt_path: str, in_channels: int, device: str) -> EncoderWithHead:
    ckpt = torch.load(ckpt_path, map_location=device)
    embed_dim = ckpt.get("embed_dim", 64)

    model = EncoderWithHead(
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_classes=2,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def extract_subject_embeddings(
    encoder_model: EncoderWithHead,
    subject,
    cfg: DatasetConfig,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    X_raw, y_raw = build_subject_windows(subject, cfg)

    if len(X_raw) == 0:
        return np.zeros((0, encoder_model.encoder.embed_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    y_bin = np.where(y_raw == 1, 0, 1).astype(np.int64)

    X_tensor = torch.from_numpy(X_raw).float().to(device)
    encoder_model.eval()
    _, Z = encoder_model(X_tensor)
    Z = Z.cpu().numpy().astype(np.float32)

    return Z, y_bin


def build_temporal_datasets_from_subjects(
    subjects,
    encoder_model: EncoderWithHead,
    cfg: DatasetConfig,
    train_subjects: List[str],
    test_subjects: List[str],
    device: str,
    seq_len: int = 4,
):
    subj_map = {s.subject_id: s for s in subjects}

    train_seq_datasets = []
    test_seq_datasets = []

    for sid in train_subjects:
        Z, y = extract_subject_embeddings(encoder_model, subj_map[sid], cfg, device)
        if len(Z) > seq_len:
            train_seq_datasets.append(TemporalSequenceDataset(Z, y, seq_len=seq_len))

    for sid in test_subjects:
        Z, y = extract_subject_embeddings(encoder_model, subj_map[sid], cfg, device)
        if len(Z) > seq_len:
            test_seq_datasets.append(TemporalSequenceDataset(Z, y, seq_len=seq_len))

    if len(train_seq_datasets) == 0 or len(test_seq_datasets) == 0:
        return None, None, None

    train_ds = ConcatSeqDataset(train_seq_datasets)
    test_ds = ConcatSeqDataset(test_seq_datasets)

    y_train_all = []
    for ds in train_seq_datasets:
        y_train_all.append(ds.y_next)
    y_train_all = np.concatenate(y_train_all, axis=0)

    return train_ds, test_ds, y_train_all


def train_one_fold(
    subjects,
    train_subjects,
    test_subjects,
    cfg,
    encoder_model,
    device,
    seq_len=4,
    hidden_dim=128,
    num_layers=2,
    num_epochs=10,
    batch_size=32,
    lr=1e-3,
):
    built = build_temporal_datasets_from_subjects(
        subjects=subjects,
        encoder_model=encoder_model,
        cfg=cfg,
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        device=device,
        seq_len=seq_len,
    )

    if built[0] is None:
        return None

    train_ds, test_ds, y_train_all = built

    if len(train_ds) == 0 or len(test_ds) == 0:
        return None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = TemporalGRU(
        embed_dim=encoder_model.encoder.embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=2,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    class_weights = compute_class_weights_from_binary_y(y_train_all, device=device)
    if class_weights is not None:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_metrics = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        train_acc, train_p, train_r, train_f1, train_cm = eval_binary_metrics(model, train_loader, device)
        test_acc, test_p, test_r, test_f1, test_cm = eval_binary_metrics(model, test_loader, device)

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_metrics = {
                "epoch": epoch,
                "train_acc": train_acc,
                "train_precision": train_p,
                "train_recall": train_r,
                "train_f1": train_f1,
                "test_acc": test_acc,
                "test_precision": test_p,
                "test_recall": test_r,
                "test_f1": test_f1,
                "test_cm": test_cm,
                "train_sequences": len(train_ds),
                "test_sequences": len(test_ds),
                "mean_train_loss": float(np.mean(losses)) if losses else 0.0,
            }

    return best_metrics


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # -----------------------------
    # SETTINGS
    # -----------------------------
    subject_ids = ["S2", "S3", "S4", "S5"]

    window_sec = 30
    stride_sec = 5
    target_rate_hz = 32
    min_label_fraction = 0.6
    channel_order = ["ECG", "EDA", "RESP", "BVP"]

    seq_len = 4
    hidden_dim = 128
    num_layers = 2
    num_epochs = 10
    batch_size = 32
    lr = 1e-3

    encoder_ckpt = "checkpoints/enc_S2-S3-S4_to_S5_win30_st5_frac0.6_ch4_d64_20260304-180513_best.pt"
    # -----------------------------

    run_name = f"loso_temporal_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join("results", "loso_temporal", run_name)
    os.makedirs(out_dir, exist_ok=True)

    cfg = DatasetConfig()
    cfg.window.window_sec = window_sec
    cfg.window.stride_sec = stride_sec
    cfg.window.target_rate_hz = target_rate_hz
    cfg.window.min_label_fraction = min_label_fraction
    cfg.channel_order = channel_order

    subjects = load_wesad_all("data/raw/WESAD", subjects=subject_ids)

    encoder_model = load_encoder_checkpoint(
        ckpt_path=encoder_ckpt,
        in_channels=len(channel_order),
        device=device,
    )
    print("Loaded encoder:", encoder_ckpt)
    print("Encoder embed_dim:", encoder_model.encoder.embed_dim)

    fold_results = []

    for test_sid in subject_ids:
        train_subjects = [s for s in subject_ids if s != test_sid]
        test_subjects = [test_sid]

        print(f"\n=== Fold: train={train_subjects} | test={test_subjects} ===")

        result = train_one_fold(
            subjects=subjects,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
            cfg=cfg,
            encoder_model=encoder_model,
            device=device,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
        )

        if result is None:
            print(f"Skipped fold for {test_sid}.")
            continue

        result["test_subject"] = test_sid
        fold_results.append(result)

        print(
            f"Best fold result | test_subject={test_sid} | "
            f"test_acc={result['test_acc']:.3f} | "
            f"test_f1={result['test_f1']:.3f} | "
            f"seq(train/test)=({result['train_sequences']}/{result['test_sequences']})"
        )

    if not fold_results:
        print("No valid folds completed.")
        return

    mean_acc = float(np.mean([r["test_acc"] for r in fold_results]))
    std_acc = float(np.std([r["test_acc"] for r in fold_results]))
    mean_f1 = float(np.mean([r["test_f1"] for r in fold_results]))
    std_f1 = float(np.std([r["test_f1"] for r in fold_results]))

    summary = {
        "run_name": run_name,
        "encoder_ckpt": encoder_ckpt,
        "subject_ids": subject_ids,
        "config": {
            "window_sec": window_sec,
            "stride_sec": stride_sec,
            "target_rate_hz": target_rate_hz,
            "min_label_fraction": min_label_fraction,
            "channel_order": channel_order,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
        "mean_test_acc": mean_acc,
        "std_test_acc": std_acc,
        "mean_test_f1": mean_f1,
        "std_test_f1": std_f1,
        "fold_results": fold_results,
    }

    out_path = os.path.join(out_dir, "loso_temporal_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LOSO TEMPORAL SUMMARY ===")
    print(f"Mean test acc: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"Mean test F1 : {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()