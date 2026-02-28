import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_loss_curve_csv(path: str):
    # expects header: epoch,train_loss,train_acc,train_f1,test_acc,test_f1
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


def plot_curves(rows, out_path: str, title: str):
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    test_f1 = [r["test_f1"] for r in rows]

    plt.figure()
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, test_f1, label="test F1")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm, out_path: str, title: str):
    # cm format: [[tn, fp],[fn, tp]]
    cm = np.asarray(cm, dtype=np.int64)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    classes = ["baseline(0)", "stress(1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=25)
    plt.yticks(tick_marks, classes)

    # annotate values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_class_balance(cm_train, cm_test, out_path: str, title: str):
    # For true counts:
    # train: baseline = tn+fp, stress = fn+tp
    cm_train = np.asarray(cm_train, dtype=np.int64)
    cm_test = np.asarray(cm_test, dtype=np.int64)

    train_baseline = int(cm_train[0, 0] + cm_train[0, 1])
    train_stress = int(cm_train[1, 0] + cm_train[1, 1])

    test_baseline = int(cm_test[0, 0] + cm_test[0, 1])
    test_stress = int(cm_test[1, 0] + cm_test[1, 1])

    labels = ["baseline", "stress"]
    x = np.arange(len(labels))

    plt.figure()
    plt.bar(x - 0.15, [train_baseline, train_stress], width=0.3, label="train")
    plt.bar(x + 0.15, [test_baseline, test_stress], width=0.3, label="test")
    plt.xticks(x, labels)
    plt.ylabel("Window count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g., results/baseline/<run_name>")
    args = ap.parse_args()

    run_dir = args.run_dir
    metrics_path = os.path.join(run_dir, "metrics.json")
    loss_path = os.path.join(run_dir, "loss_curve.csv")
    cm_test_path = os.path.join(run_dir, "confusion_matrix_test.npy")
    cm_train_path = os.path.join(run_dir, "confusion_matrix_train.npy")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(metrics_path)
    if not os.path.exists(loss_path):
        raise FileNotFoundError(loss_path)
    if not os.path.exists(cm_test_path):
        raise FileNotFoundError(cm_test_path)
    if not os.path.exists(cm_train_path):
        raise FileNotFoundError(cm_train_path)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    rows = read_loss_curve_csv(loss_path)
    cm_test = np.load(cm_test_path)
    cm_train = np.load(cm_train_path)

    run_name = metrics.get("run_name", os.path.basename(run_dir))
    cfg = metrics.get("config", {})
    subtitle = (
        f"train={cfg.get('train_subjects')} test={cfg.get('test_subjects')} "
        f"win={cfg.get('window_sec')}s stride={cfg.get('stride_sec')}s "
        f"min_frac={cfg.get('min_label_fraction')} channels={len(cfg.get('channel_order', []))}"
    )

    out1 = os.path.join(run_dir, "plot_curves.png")
    out2 = os.path.join(run_dir, "plot_confusion_test.png")
    out3 = os.path.join(run_dir, "plot_class_balance.png")

    plot_curves(rows, out1, title=f"{run_name}\n{subtitle}")
    plot_confusion_matrix(cm_test, out2, title=f"Test Confusion Matrix\n{run_name}")
    plot_class_balance(cm_train, cm_test, out3, title=f"Class Balance (True Counts)\n{run_name}")

    print("Saved:")
    print(" ", out1)
    print(" ", out2)
    print(" ", out3)


if __name__ == "__main__":
    main()