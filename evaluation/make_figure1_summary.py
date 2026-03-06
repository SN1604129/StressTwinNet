import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_loss_curve_csv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            row = {k: float(v) for k, v in zip(header, parts)}
            rows.append(row)
    return rows


def plot_confusion(ax, cm, title: str):
    cm = np.asarray(cm, dtype=np.int64)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["baseline(0)", "stress(1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, rotation=20)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")


def plot_curves(ax, rows, title: str):
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    test_f1 = [r["test_f1"] for r in rows]

    ax.plot(epochs, train_loss, label="train loss")
    ax.plot(epochs, test_f1, label="test F1")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()


def load_image_as_array(path: str):
    # matplotlib can read png directly
    img = plt.imread(path)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_run_dir", required=True, help="e.g., results/baseline/<run_name>")
    ap.add_argument("--encoder_run_dir", required=True, help="e.g., results/embeddings/<run_name>")
    ap.add_argument("--out", default="results/summary/figure1_baseline_encoder_summary.png")
    args = ap.parse_args()

    baseline_dir = args.baseline_run_dir
    encoder_dir = args.encoder_run_dir

    # --- baseline files ---
    metrics_path = os.path.join(baseline_dir, "metrics.json")
    loss_path = os.path.join(baseline_dir, "loss_curve.csv")
    cm_test_path = os.path.join(baseline_dir, "confusion_matrix_test.npy")

    for p in [metrics_path, loss_path, cm_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        base_metrics = json.load(f)

    rows = read_loss_curve_csv(loss_path)
    cm_test = np.load(cm_test_path)

    base_run_name = base_metrics.get("run_name", os.path.basename(baseline_dir))
    base_cfg = base_metrics.get("config", {})

    base_subtitle = (
        f"train={base_cfg.get('train_subjects')} test={base_cfg.get('test_subjects')} | "
        f"win={base_cfg.get('window_sec')}s stride={base_cfg.get('stride_sec')}s | "
        f"min_frac={base_cfg.get('min_label_fraction')}"
    )

    # --- encoder PCA plot ---
    pca_path = os.path.join(encoder_dir, "pca_test.png")
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"Missing: {pca_path}")

    pca_img = load_image_as_array(pca_path)
    enc_run_name = os.path.basename(encoder_dir)

    # --- make output dir ---
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # --- Figure layout (3 panels) ---
    fig = plt.figure(figsize=(14, 4.5))

    # Panel A: curves
    ax1 = fig.add_subplot(1, 3, 1)
    plot_curves(ax1, rows, title="A) Baseline training curves")

    # Panel B: confusion matrix
    ax2 = fig.add_subplot(1, 3, 2)
    plot_confusion(ax2, cm_test, title="B) Baseline test confusion matrix")

    # Panel C: encoder PCA image
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(pca_img)
    ax3.axis("off")
    ax3.set_title("C) Encoder latent space (PCA, test)")

    # Overall title
    fig.suptitle(
        "StressTwinNet — Baseline + Latent Encoder Summary\n"
        f"Baseline: {base_run_name}\n{base_subtitle}\n"
        f"Encoder: {enc_run_name}",
        y=1.06,
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print("Saved summary figure to:", out_path)


if __name__ == "__main__":
    main()