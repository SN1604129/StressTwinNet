import os
import json
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.data_loader import load_wesad_all
from utils.dataset import DatasetConfig, StressWindowDataset, build_subject_splits
from models.cnn_encoder import EncoderWithHead


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def eval_binary_metrics(model, loader, device: str):
    model.eval()
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
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


@torch.no_grad()
def collect_embeddings(model, loader, device: str):
    model.eval()
    Z, Y = [], []
    for x, y in loader:
        x = x.to(device)
        logits, z = model(x)
        Z.append(z.cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(Z, axis=0), np.concatenate(Y, axis=0)


def pca_2d(Z: np.ndarray):
    Zc = Z - Z.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Zc @ Vt[:2].T


def save_pca_plot(Z: np.ndarray, y: np.ndarray, out_path: str, title: str):
    Z2 = pca_2d(Z)
    plt.figure()
    for cls, name in [(0, "baseline"), (1, "stress")]:
        idx = (y == cls)
        plt.scatter(Z2[idx, 0], Z2[idx, 1], s=8, label=name)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_one_fold(subjects, train_subjects, test_subjects, cfg, device, embed_dim=64, num_epochs=10, batch_size=16, lr=1e-3):
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

    model = EncoderWithHead(
        in_channels=len(cfg.channel_order),
        embed_dim=embed_dim,
        num_classes=2,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    cw = compute_class_weights_from_raw_y(ytr, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=cw) if cw is not None else torch.nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    best_metrics = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        tr_acc, tr_p, tr_r, tr_f1, tr_cm = eval_binary_metrics(model, train_loader, device)
        te_acc, te_p, te_r, te_f1, te_cm = eval_binary_metrics(model, test_loader, device)

        if te_f1 > best_f1:
            best_f1 = te_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
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
            }

    model.load_state_dict(best_state)

    Ztr, Ytr = collect_embeddings(model, train_loader, device)
    Zte, Yte = collect_embeddings(model, test_loader, device)

    return {
        "best_metrics": best_metrics,
        "model_state": best_state,
        "train_embeddings": (Ztr, Ytr),
        "test_embeddings": (Zte, Yte),
    }


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

    embed_dim = 64
    num_epochs = 10
    batch_size = 16
    lr = 1e-3

    run_name = f"loso_encoder_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join("results", "loso_encoder", run_name)
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
            embed_dim=embed_dim,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
        )

        if result is None:
            print(f"Skipped fold for {test_sid}.")
            continue

        metrics = result["best_metrics"]
        metrics["test_subject"] = test_sid
        fold_results.append(metrics)

        fold_dir = os.path.join(out_dir, f"fold_{test_sid}")
        os.makedirs(fold_dir, exist_ok=True)

        Ztr, Ytr = result["train_embeddings"]
        Zte, Yte = result["test_embeddings"]

        np.savez(os.path.join(fold_dir, "train_embeddings.npz"), Z=Ztr, y=Ytr)
        np.savez(os.path.join(fold_dir, "test_embeddings.npz"), Z=Zte, y=Yte)
        save_pca_plot(Zte, Yte, os.path.join(fold_dir, "pca_test.png"), title=f"PCA Test Fold {test_sid}")

        print(
            f"Best fold result | test_subject={test_sid} | "
            f"test_acc={metrics['test_acc']:.3f} | test_f1={metrics['test_f1']:.3f}"
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
            "embed_dim": embed_dim,
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

    with open(os.path.join(out_dir, "loso_encoder_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LOSO ENCODER SUMMARY ===")
    print(f"Mean test acc: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"Mean test F1 : {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"Saved to: {os.path.join(out_dir, 'loso_encoder_summary.json')}")


if __name__ == "__main__":
    main()