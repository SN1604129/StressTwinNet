from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from torch.utils.data import DataLoader

from models.attention_temporal import AttentionTemporalStressTwinNet
from utils.sequence_dataset import StressSequenceDataset


def evaluate(model, loader, device):
    model.eval()

    all_y = []
    all_pred = []
    all_attn = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)  # [B, S, M, T]
            y = y.to(device)

            logits, attn = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

            pred = torch.argmax(logits, dim=1)

            all_y.extend(y.cpu().numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())
            all_attn.append(attn.cpu().numpy())

    all_y = np.array(all_y)
    all_pred = np.array(all_pred)
    all_attn = np.concatenate(all_attn, axis=0) if len(all_attn) > 0 else None

    metrics = {
        "loss": total_loss / max(len(loader), 1),
        "acc": accuracy_score(all_y, all_pred),
        "f1": f1_score(all_y, all_pred, zero_division=0),
        "precision": precision_score(all_y, all_pred, zero_division=0),
        "recall": recall_score(all_y, all_pred, zero_division=0),
        "cm": confusion_matrix(all_y, all_pred),
        "attn_mean": all_attn.mean(axis=0) if all_attn is not None else None,
    }
    return metrics


def train_one_run(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    train_subjects: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    test_subjects: np.ndarray,
    sequence_length: int = 4,
    batch_size: int = 64,
    epochs: int = 25,
    lr: float = 1e-3,
    latent_dim: int = 64,
    gru_hidden_dim: int = 128,
    checkpoint_dir: str = "checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = StressSequenceDataset(
        train_windows, train_labels, train_subjects, sequence_length=sequence_length
    )
    test_ds = StressSequenceDataset(
        test_windows, test_labels, test_subjects, sequence_length=sequence_length
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    modality_names = ["ECG", "EDA", "RESP", "BVP"]

    model = AttentionTemporalStressTwinNet(
        modality_names=modality_names,
        latent_dim=latent_dim,
        gru_hidden_dim=gru_hidden_dim,
        gru_layers=1,
        num_classes=2,
        dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_f1 = -1.0
    best_path = None

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_name = time.strftime("attention_temporal_%Y%m%d-%H%M%S")
    best_path = str(Path(checkpoint_dir) / f"{run_name}_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"loss {running_loss / max(len(train_loader),1):.4f} | "
            f"train acc {train_metrics['acc']:.3f} f1 {train_metrics['f1']:.3f} | "
            f"test acc {test_metrics['acc']:.3f} f1 {test_metrics['f1']:.3f}"
        )
        print(
            f"  Train CM [[tn, fp],[fn, tp]] = {train_metrics['cm'].tolist()} | "
            f"P={train_metrics['precision']:.3f} R={train_metrics['recall']:.3f}"
        )
        print(
            f"  Test  CM [[tn, fp],[fn, tp]] = {test_metrics['cm'].tolist()} | "
            f"P={test_metrics['precision']:.3f} R={test_metrics['recall']:.3f}"
        )

        if test_metrics["attn_mean"] is not None:
            print(f"  Test mean attention = {test_metrics['attn_mean'].tolist()}")

        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_f1": best_f1,
                    "modality_names": modality_names,
                    "sequence_length": sequence_length,
                    "latent_dim": latent_dim,
                    "gru_hidden_dim": gru_hidden_dim,
                },
                best_path,
            )
            print(f"  Best so far: {best_path}")

    print("\nTraining complete.")
    print("Best checkpoint:", best_path)
    print("Best test F1:", round(best_f1, 4))