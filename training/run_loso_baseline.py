import os
import json
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_loader import load_wesad_all
from utils.dataset import DatasetConfig, StressWindowDataset, build_subject_splits
from models.cnn_baseline import CNNBaseline


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def compute_class_weights_from_raw_y(y_raw: np.ndarray, device: str):
    if y_raw.size == 0:
        return None

    y_bin = np.where(y_raw == 1, 0, 1).astype(np.int64)
    n0 = int((y_bin == 0).sum())
    n1 = int((y_bin == 1).sum())

    if n0 == 0 or n1 == 0:
        return None

    w0 = (n0 + n1) / (2.0 * n0)
    w1 = (n0 + n1) / (2.0 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def train_one_fold(subjects, train_subjects, test_subjects, cfg, device, num_epochs=10, batch_size=16, lr=1e-3):
    (Xtr, ytr), (Xte, yte) = build_subject_splits(
        subjects,
        train_subject_ids=train_subjects,
        test_subject_ids=test_subjects,
        cfg=cfg,
    )

    if ytr.size == 0 or yte.size == 0:
        return None

    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None

    train_ds = StressWindowDataset.from_arrays(Xtr, ytr, cfg=cfg)
    test_ds = StressWindowDataset.from_arrays(Xte, yte, cfg=cfg)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = CNNBaseline(in_channels=len(cfg.channel_order), num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    cw = compute_class_weights_from_raw_y(ytr, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=cw) if cw is not None else torch.nn.CrossEntropyLoss()

    best = None
    best_f1 = -1.0

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

        tr_acc, tr_p, tr_r, tr_f1, tr_cm = eval_binary_metrics(model, train_loader, device)
        te_acc, te_p, te_r, te_f1, te_cm = eval_binary_metrics(model, test_loader, device)

        if te_f1 > best_f1:
            best_f1 = te_f1
            best = {
                "epoch": epoch,
                "train_acc": tr_acc,
                "train_precision": tr_p,
                "train_recall": tr_r,
                "train_f1": tr_f1,
                "test_acc": te_acc,
                "test_precision": te_p,
                "test_recall": te_r,
                "test_f1": te_f1,
                "test_cm": te_cm,
                "train_windows": len(train_ds),
                "test_windows": len(test_ds),
                "mean_train_loss": float(np.mean(losses)) if losses else 0.0,
            }

    return best


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    subject_ids = ["S2", "S3", "S4", "S5"]
    subjects = load_wesad_all("data/raw/WESAD", subjects=subject_ids)

    cfg = DatasetConfig()
    cfg.window.window_sec = 30
    cfg.window.stride_sec = 5
    cfg.window.target_rate_hz = 32
    cfg.window.min_label_fraction = 0.6
    cfg.channel_order = ["ECG", "EDA", "RESP", "BVP"]

    run_name = f"loso_baseline_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join("results", "loso", run_name)
    os.makedirs(out_dir, exist_ok=True)

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
            device=device,
            num_epochs=10,
            batch_size=16,
            lr=1e-3,
        )

        if result is None:
            print(f"Skipped fold for {test_sid} (insufficient valid windows/classes).")
            continue

        result["test_subject"] = test_sid
        fold_results.append(result)

        print(
            f"Best fold result | "
            f"test_acc={result['test_acc']:.3f} | "
            f"test_f1={result['test_f1']:.3f} | "
            f"windows(train/test)=({result['train_windows']}/{result['test_windows']})"
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
        "subject_ids": subject_ids,
        "config": {
            "window_sec": cfg.window.window_sec,
            "stride_sec": cfg.window.stride_sec,
            "target_rate_hz": cfg.window.target_rate_hz,
            "min_label_fraction": cfg.window.min_label_fraction,
            "channel_order": cfg.channel_order,
        },
        "mean_test_acc": mean_acc,
        "std_test_acc": std_acc,
        "mean_test_f1": mean_f1,
        "std_test_f1": std_f1,
        "fold_results": fold_results,
    }

    out_path = os.path.join(out_dir, "loso_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LOSO SUMMARY ===")
    print(f"Mean test acc: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"Mean test F1 : {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()