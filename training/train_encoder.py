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


def eval_binary_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor):
    preds = torch.argmax(logits, dim=1)
    tp = int(((preds == 1) & (y == 1)).sum().item())
    tn = int(((preds == 0) & (y == 0)).sum().item())
    fp = int(((preds == 1) & (y == 0)).sum().item())
    fn = int(((preds == 0) & (y == 1)).sum().item())

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
    Z = []
    Y = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, z = model(x)
        Z.append(z.cpu().numpy())
        Y.append(y.cpu().numpy())
    Z = np.concatenate(Z, axis=0)
    Y = np.concatenate(Y, axis=0)
    return Z, Y


def pca_2d(Z: np.ndarray):
    # simple PCA using SVD (no sklearn)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    Z2 = Zc @ Vt[:2].T
    return Z2


def save_pca_plot(Z: np.ndarray, y: np.ndarray, out_path: str, title: str):
    Z2 = pca_2d(Z)
    plt.figure()
    for cls, name in [(0, "baseline"), (1, "stress")]:
        idx = (y == cls)
        plt.scatter(Z2[idx, 0], Z2[idx, 1], s=8, label=name)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
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
    min_label_fraction = 0.6

    channel_order = ["ECG", "EDA", "RESP", "BVP"]

    embed_dim = 64          # try 32/64/128
    num_epochs = 20
    batch_size = 16
    lr = 1e-3
    # -----------------------------

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"enc_{'-'.join(train_subjects)}_to_{'-'.join(test_subjects)}_"
        f"win{window_sec}_st{stride_sec}_frac{min_label_fraction}_"
        f"ch{len(channel_order)}_d{embed_dim}_{ts}"
    )

    os.makedirs("checkpoints", exist_ok=True)
    out_dir = os.path.join("results", "embeddings", run_name)
    os.makedirs(out_dir, exist_ok=True)

    print("Run:", run_name)
    print("Saving to:", out_dir)

    subjects = load_wesad_all("data/raw/WESAD", subjects=subjects_to_load)

    cfg = DatasetConfig()
    cfg.window.window_sec = window_sec
    cfg.window.stride_sec = stride_sec
    cfg.window.target_rate_hz = target_rate_hz
    cfg.window.min_label_fraction = min_label_fraction
    cfg.channel_order = channel_order

    (Xtr, ytr), (Xte, yte) = build_subject_splits(
        subjects,
        train_subject_ids=train_subjects,
        test_subject_ids=test_subjects,
        cfg=cfg,
    )

    train_ds = StressWindowDataset.from_arrays(Xtr, ytr, cfg=cfg)
    test_ds = StressWindowDataset.from_arrays(Xte, yte, cfg=cfg)

    print(f"Train windows: {len(train_ds)} | Test windows: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = EncoderWithHead(in_channels=len(channel_order), embed_dim=embed_dim, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    cw = compute_class_weights_from_raw_y(ytr, device=device)
    if cw is not None:
        print(f"Class weights: baseline={cw[0].item():.3f}, stress={cw[1].item():.3f}")
        loss_fn = torch.nn.CrossEntropyLoss(weight=cw)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_ckpt = os.path.join("checkpoints", f"{run_name}_best.pt")

    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits, z = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # eval quick on full loaders
        model.eval()
        # compute cm/f1 over loaders
        def eval_loader(loader):
            t_tp = t_tn = t_fp = t_fn = 0
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    y = y.to(device)
                    logits, _ = model(x)
                    preds = torch.argmax(logits, dim=1)
                    t_tp += int(((preds == 1) & (y == 1)).sum().item())
                    t_tn += int(((preds == 0) & (y == 0)).sum().item())
                    t_fp += int(((preds == 1) & (y == 0)).sum().item())
                    t_fn += int(((preds == 0) & (y == 1)).sum().item())

            total = t_tp + t_tn + t_fp + t_fn
            acc = (t_tp + t_tn) / total if total > 0 else 0.0
            precision = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0
            recall = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            cm = [[t_tn, t_fp], [t_fn, t_tp]]
            return acc, precision, recall, f1, cm

        tr_acc, tr_p, tr_r, tr_f1, tr_cm = eval_loader(train_loader)
        te_acc, te_p, te_r, te_f1, te_cm = eval_loader(test_loader)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"Epoch {epoch:02d} | loss {mean_loss:.4f} | "
            f"train acc {tr_acc:.3f} f1 {tr_f1:.3f} | "
            f"test acc {te_acc:.3f} f1 {te_f1:.3f}"
        )

        if te_f1 > best_f1:
            best_f1 = te_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "run_name": run_name,
                    "embed_dim": embed_dim,
                    "channel_order": channel_order,
                    "cfg": {
                        "train_subjects": train_subjects,
                        "test_subjects": test_subjects,
                        "window_sec": window_sec,
                        "stride_sec": stride_sec,
                        "target_rate_hz": target_rate_hz,
                        "min_label_fraction": min_label_fraction,
                    },
                    "best_test_f1": float(best_f1),
                    "test_cm": te_cm,
                },
                best_ckpt,
            )

    print(f"Best test F1={best_f1:.3f} | checkpoint={best_ckpt}")

    # Collect embeddings using best model weights (load)
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    Ztr, Ytr = collect_embeddings(model, train_loader, device)
    Zte, Yte = collect_embeddings(model, test_loader, device)

    np.savez(os.path.join(out_dir, "train_embeddings.npz"), Z=Ztr, y=Ytr)
    np.savez(os.path.join(out_dir, "test_embeddings.npz"), Z=Zte, y=Yte)

    save_pca_plot(Zte, Yte, os.path.join(out_dir, "pca_test.png"), title=f"PCA (test) {run_name}")

    # Save run metadata
    meta = {
        "run_name": run_name,
        "best_checkpoint": best_ckpt,
        "best_test_f1": float(best_f1),
        "train_windows": int(Ztr.shape[0]),
        "test_windows": int(Zte.shape[0]),
        "embed_dim": embed_dim,
        "channel_order": channel_order,
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved embeddings to:", out_dir)


if __name__ == "__main__":
    main()