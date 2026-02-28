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


def eval_binary_metrics(model: torch.nn.Module, loader: DataLoader, device: str):
    """
    Binary metrics where positive class is stress (1).
    Confusion matrix format: [[tn, fp],
                              [fn, tp]]
    Returns: acc, cm, precision, recall, f1
    """
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
    return acc, cm, precision, recall, f1


def compute_class_weights_from_raw_y(y_raw: np.ndarray, device: str):
    """
    raw: 1=baseline, 2=stress
    bin: 0=baseline, 1=stress
    Returns torch tensor [w0, w1] or None if one class missing.
    """
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


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_loss_curve_csv(path: str, rows: list[dict]) -> None:
    # Simple CSV writer without pandas
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


def main() -> None:
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # -----------------------------
    # EXPERIMENT CONTROLS
    # -----------------------------
    subjects_to_load = ["S2", "S3", "S4", "S5"]
    train_subjects = ["S2", "S3", "S4"]
    test_subjects = ["S5"]

    window_sec = 30
    stride_sec = 5
    target_rate_hz = 32
    min_label_fraction = 0.9

    # channel_order = ["ECG", "EDA", "RESP", "BVP"]
    channel_order = ["ECG", "EDA", "RESP"]  # chest-only
    # channel_order = ["BVP"]                 # wrist-only

    num_epochs = 20
    batch_size = 8
    lr = 1e-3
    # -----------------------------

    # Run name (unique-ish, but readable)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"cnn_{'-'.join(train_subjects)}_to_{'-'.join(test_subjects)}_"
        f"win{window_sec}_st{stride_sec}_frac{min_label_fraction}_"
        f"ch{len(channel_order)}_{ts}"
    )

    # Output dirs
    os.makedirs("checkpoints", exist_ok=True)
    run_dir = os.path.join("results", "baseline", run_name)
    os.makedirs(run_dir, exist_ok=True)

    print("Run:", run_name)
    print("Saving to:", run_dir)

    # Load data
    subjects = load_wesad_all("data/raw/WESAD", subjects=subjects_to_load)

    # Config
    cfg = DatasetConfig()
    cfg.window.window_sec = window_sec
    cfg.window.stride_sec = stride_sec
    cfg.window.target_rate_hz = target_rate_hz
    cfg.window.min_label_fraction = min_label_fraction
    cfg.channel_order = channel_order

    print(
        "Config:",
        f"train={train_subjects} | test={test_subjects} | "
        f"channels={cfg.channel_order} | window={cfg.window.window_sec}s | "
        f"stride={cfg.window.stride_sec}s | rate={cfg.window.target_rate_hz}Hz | "
        f"min_frac={cfg.window.min_label_fraction}"
    )

    # Split by subject (no leakage)
    (Xtr, ytr), (Xte, yte) = build_subject_splits(
        subjects,
        train_subject_ids=train_subjects,
        test_subject_ids=test_subjects,
        cfg=cfg,
    )

    if ytr.size == 0 or yte.size == 0:
        print("ERROR: No windows produced. Check windowing config.")
        return

    if len(np.unique(ytr)) < 2:
        print("ERROR: Train windows contain only one class. Fix windowing/labels before training.")
        return

    train_ds = StressWindowDataset.from_arrays(Xtr, ytr, cfg=cfg)
    test_ds = StressWindowDataset.from_arrays(Xte, yte, cfg=cfg)

    print(f"Train windows: {len(train_ds)} | Test windows: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = CNNBaseline(in_channels=len(cfg.channel_order), num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss + optional class weights
    class_weights = compute_class_weights_from_raw_y(ytr, device=device)
    if class_weights is not None:
        print(f"Class weights: baseline={class_weights[0].item():.3f}, stress={class_weights[1].item():.3f}")
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("Class weights skipped.")
        loss_fn = torch.nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_ckpt_path = os.path.join("checkpoints", f"{run_name}_best.pt")

    loss_rows = []
    best_snapshot = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        train_acc, train_cm, train_p, train_r, train_f1 = eval_binary_metrics(model, train_loader, device)
        test_acc, test_cm, test_p, test_r, test_f1 = eval_binary_metrics(model, test_loader, device)

        mean_loss = float(np.mean(train_losses)) if train_losses else 0.0
        loss_rows.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_acc": float(train_acc),
                "train_f1": float(train_f1),
                "test_acc": float(test_acc),
                "test_f1": float(test_f1),
            }
        )

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_snapshot = {
                "epoch": epoch,
                "train_cm": train_cm,
                "test_cm": test_cm,
                "train_precision": float(train_p),
                "train_recall": float(train_r),
                "train_f1": float(train_f1),
                "test_precision": float(test_p),
                "test_recall": float(test_r),
                "test_f1": float(test_f1),
                "test_acc": float(test_acc),
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "run_name": run_name,
                    "cfg": {
                        "subjects_to_load": subjects_to_load,
                        "train_subjects": train_subjects,
                        "test_subjects": test_subjects,
                        "channel_order": cfg.channel_order,
                        "window_sec": cfg.window.window_sec,
                        "stride_sec": cfg.window.stride_sec,
                        "target_rate_hz": cfg.window.target_rate_hz,
                        "min_label_fraction": cfg.window.min_label_fraction,
                        "batch_size": batch_size,
                        "lr": lr,
                        "num_epochs": num_epochs,
                    },
                    "best_test_f1": float(best_f1),
                },
                best_ckpt_path,
            )

        print(
            f"Epoch {epoch:02d} | loss {mean_loss:.4f} | "
            f"train acc {train_acc:.3f} f1 {train_f1:.3f} | "
            f"test acc {test_acc:.3f} f1 {test_f1:.3f}"
        )

        if epoch in (1, 5, 10, num_epochs):
            print(f"  Train CM [[tn, fp],[fn, tp]] = {train_cm} | P={train_p:.3f} R={train_r:.3f}")
            print(f"  Test  CM [[tn, fp],[fn, tp]] = {test_cm} | P={test_p:.3f} R={test_r:.3f}")
            print(f"  Best so far: {best_ckpt_path} (best test F1={best_f1:.3f})")

    # Save results
    np.save(os.path.join(run_dir, "confusion_matrix_test.npy"), np.array(best_snapshot["test_cm"], dtype=np.int64))
    np.save(os.path.join(run_dir, "confusion_matrix_train.npy"), np.array(best_snapshot["train_cm"], dtype=np.int64))
    save_loss_curve_csv(os.path.join(run_dir, "loss_curve.csv"), loss_rows)

    metrics = {
        "run_name": run_name,
        "device": device,
        "best_checkpoint": best_ckpt_path,
        "best_test_f1": float(best_f1),
        "best_snapshot": best_snapshot,
        "config": {
            "subjects_to_load": subjects_to_load,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "channel_order": cfg.channel_order,
            "window_sec": cfg.window.window_sec,
            "stride_sec": cfg.window.stride_sec,
            "target_rate_hz": cfg.window.target_rate_hz,
            "min_label_fraction": cfg.window.min_label_fraction,
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
        },
        "data": {
            "train_windows": int(len(train_ds)),
            "test_windows": int(len(test_ds)),
        },
    }
    save_json(os.path.join(run_dir, "metrics.json"), metrics)

    print("Done.")
    print(f"Saved run results to: {run_dir}")
    print(f"Best model saved to: {best_ckpt_path} (best test F1={best_f1:.3f})")


if __name__ == "__main__":
    main()