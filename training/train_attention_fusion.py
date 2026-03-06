import os
import json
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_loader import load_wesad_all
from utils.dataset import DatasetConfig, StressWindowDataset, build_subject_splits
from models.stresstwinnet import StressTwinNetAttention


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval_binary_metrics(model, loader, device: str):
    model.eval()
    tp = tn = fp = fn = 0
    attn_all = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits, _, attn = model(x)
            preds = torch.argmax(logits, dim=1)

            tp += int(((preds == 1) & (y == 1)).sum().item())
            tn += int(((preds == 0) & (y == 0)).sum().item())
            fp += int(((preds == 1) & (y == 0)).sum().item())
            fn += int(((preds == 0) & (y == 1)).sum().item())

            attn_all.append(attn.cpu().numpy())

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    cm = [[tn, fp], [fn, tp]]

    attn_all = np.concatenate(attn_all, axis=0) if len(attn_all) > 0 else None
    mean_attn = attn_all.mean(axis=0).tolist() if attn_all is not None else None

    return acc, precision, recall, f1, cm, mean_attn


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


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # -----------------------------
    # SETTINGS
    # -----------------------------
    subjects_to_load = ["S2", "S3", "S4", "S5"]
    train_subjects = ["S2", "S3", "S4"]
    test_subjects = ["S5"]

    window_sec = 30
    stride_sec = 5
    target_rate_hz = 32
    min_label_fraction = 0.6
    channel_order = ["ECG", "EDA", "RESP", "BVP"]

    embed_dim = 64
    num_epochs = 20
    batch_size = 16
    lr = 1e-3
    # -----------------------------

    run_name = f"attention_fusion_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join("results", "attention", run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    print("Run:", run_name)
    print("Saving to:", out_dir)

    cfg = DatasetConfig()
    cfg.window.window_sec = window_sec
    cfg.window.stride_sec = stride_sec
    cfg.window.target_rate_hz = target_rate_hz
    cfg.window.min_label_fraction = min_label_fraction
    cfg.channel_order = channel_order

    subjects = load_wesad_all("data/raw/WESAD", subjects=subjects_to_load)

    (Xtr, ytr), (Xte, yte) = build_subject_splits(
        subjects,
        train_subject_ids=train_subjects,
        test_subject_ids=test_subjects,
        cfg=cfg,
    )

    train_ds = StressWindowDataset.from_arrays(Xtr, ytr, cfg=cfg)
    test_ds = StressWindowDataset.from_arrays(Xte, yte, cfg=cfg)

    print(f"Train windows: {len(train_ds)}")
    print(f"Test windows : {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = StressTwinNetAttention(embed_dim=embed_dim, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    class_weights = compute_class_weights_from_raw_y(ytr, device=device)
    if class_weights is not None:
        print(f"Class weights: baseline={class_weights[0].item():.3f}, stress={class_weights[1].item():.3f}")
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_ckpt = os.path.join("checkpoints", f"{run_name}_best.pt")
    history = []
    best_attention = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits, _, _ = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        train_acc, train_p, train_r, train_f1, train_cm, train_attn = eval_binary_metrics(model, train_loader, device)
        test_acc, test_p, test_r, test_f1, test_cm, test_attn = eval_binary_metrics(model, test_loader, device)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        history.append(
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
            best_attention = {
                "train_mean_attention": train_attn,
                "test_mean_attention": test_attn,
                "modality_order": channel_order,
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "run_name": run_name,
                    "config": {
                        "subjects_to_load": subjects_to_load,
                        "train_subjects": train_subjects,
                        "test_subjects": test_subjects,
                        "channel_order": channel_order,
                        "window_sec": window_sec,
                        "stride_sec": stride_sec,
                        "target_rate_hz": target_rate_hz,
                        "min_label_fraction": min_label_fraction,
                        "embed_dim": embed_dim,
                        "batch_size": batch_size,
                        "lr": lr,
                        "num_epochs": num_epochs,
                    },
                    "best_test_f1": float(best_f1),
                    "best_test_cm": test_cm,
                    "best_attention": best_attention,
                },
                best_ckpt,
            )

        print(
            f"Epoch {epoch:02d} | "
            f"loss {mean_loss:.4f} | "
            f"train acc {train_acc:.3f} f1 {train_f1:.3f} | "
            f"test acc {test_acc:.3f} f1 {test_f1:.3f}"
        )

        if epoch in (1, 5, 10, num_epochs):
            print(f"  Train CM [[tn, fp],[fn, tp]] = {train_cm} | P={train_p:.3f} R={train_r:.3f}")
            print(f"  Test  CM [[tn, fp],[fn, tp]] = {test_cm} | P={test_p:.3f} R={test_r:.3f}")
            print(f"  Test mean attention = {test_attn}")
            print(f"  Best so far: {best_ckpt} (best test F1={best_f1:.3f})")

    summary = {
        "run_name": run_name,
        "best_ckpt": best_ckpt,
        "best_test_f1": float(best_f1),
        "best_attention": best_attention,
        "train_windows": int(len(train_ds)),
        "test_windows": int(len(test_ds)),
        "history": history,
    }

    with open(os.path.join(out_dir, "attention_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Saved summary to: {os.path.join(out_dir, 'attention_summary.json')}")
    print(f"Best model: {best_ckpt} | best test F1={best_f1:.3f}")
    print(f"Best attention: {best_attention}")


if __name__ == "__main__":
    main()