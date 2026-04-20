"""
Smart Posture Shirt - Production Training Script
- Framework: PyTorch
- Deployment Target: ESP32-C6 (TFLite Micro)
- Features: INT8 Quantization, tqdm Progress Bar, English Dashboard
- Local Version: No Colab dependencies, read dataset from local ./dataset folder
"""
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm.auto import tqdm

# ════════════════════════════════════════════════════
# 1. Global Configurations (Local Path Modified)
# ════════════════════════════════════════════════════
SAMPLE_RATE = 50
WINDOW_SEC  = 2
WIN         = SAMPLE_RATE * WINDOW_SEC   # 100
STRIDE      = WIN // 2                   # 50
N_SENSORS   = 9
N_CLASSES   = 4

# Labels updated as requested
CLASS_NAMES = ["Upright", "Hunchback", "Left Side Bend", "Right Side Bend"]

SENSOR_COLS = ["fsr_left", "fsr_right", "fsr_spine", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
LABEL_COL, TIME_COL = "label", "time"

# ==============================================
# ✅ 本地路径修改（核心：删除Colab路径）
# ==============================================
CSV_FOLDER = "./dataset"    # 数据集放在脚本同级的dataset文件夹
OUT_DIR    = "./output"     # 输出模型/图表到本地output文件夹
PLOT_DIR = os.path.join(OUT_DIR, "plots")
COLORS   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# Hyperparameters
EPOCHS, BATCH_SIZE = 60, 128
LR, PATIENCE = 5e-4, 12
LR_PATIENCE, LR_FACTOR, LR_MIN = 6, 0.5, 1e-6

# ════════════════════════════════════════════════════
# 2. Data Processing
# ════════════════════════════════════════════════════
def load_data(folder):
    files = glob.glob(os.path.join(folder, "**/*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No CSV in {folder}")
    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        if all(c in df.columns for c in SENSOR_COLS + [LABEL_COL]):
            dfs.append(df[SENSOR_COLS + [LABEL_COL]])
    df_all = pd.concat(dfs, ignore_index=True).dropna()
    X = df_all[SENSOR_COLS].values.astype(np.float32)
    y = df_all[LABEL_COL].values.astype(np.int64)
    print(f"Total Rows Loaded: {len(X):,}")
    return X, y

def sliding_window(X, y):
    Xs, ys = [], []
    for i in range(0, len(X) - WIN, STRIDE):
        Xs.append(X[i:i + WIN])
        ys.append(np.bincount(y[i:i + WIN]).argmax())
    return np.array(Xs), np.array(ys)

# ════════════════════════════════════════════════════
# 3. Model & Training
# ════════════════════════════════════════════════════
class Posture1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(N_SENSORS, 16, 5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool, self.relu = nn.MaxPool1d(2), nn.ReLU()
        self.gap, self.fc = nn.AdaptiveAvgPool1d(1), nn.Linear(64, 32)
        self.drop, self.out = nn.Dropout(0.3), nn.Linear(32, N_CLASSES)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.gap(x).squeeze(-1)
        x = self.relu(self.fc(x))
        x = self.drop(x)
        return self.out(x)

def train_model(X_tr, y_tr, X_val, y_val, device):
    model = Posture1DCNN().to(device)
    tr_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                       batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                        batch_size=BATCH_SIZE, num_workers=0)

    weights = 1.0 / np.bincount(y_tr)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights/weights.sum()*N_CLASSES,
                                                       dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=LR_PATIENCE,
                                                     factor=LR_FACTOR, min_lr=LR_MIN)

    history = {"loss":[], "accuracy":[], "val_loss":[], "val_accuracy":[]}
    best_acc, best_state, no_improve = 0, None, 0

    print(f"Training on {device}...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss, t_corr, t_total = 0, 0, 0
        pbar = tqdm(tr_dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False, ncols=100)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(yb)
            t_corr += (logits.argmax(1)==yb).sum().item()
            t_total += len(yb)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        model.eval()
        v_loss, v_corr, v_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss += criterion(logits, yb).item() * len(yb)
                v_corr += (logits.argmax(1)==yb).sum().item()
                v_total += len(yb)

        history["loss"].append(t_loss/t_total)
        history["accuracy"].append(t_corr/t_total)
        history["val_loss"].append(v_loss/v_total)
        history["val_accuracy"].append(v_corr/v_total)
        scheduler.step(v_loss/v_total)
        print(f"Epoch {epoch:02d} - loss: {t_loss/t_total:.4f} - acc: {t_corr/t_total:.4f} - val_loss: {v_loss/v_total:.4f} - val_acc: {v_corr/v_total:.4f}")

        if (v_corr/v_total) > best_acc:
            best_acc, best_state, no_improve = (v_corr/v_total), model.state_dict().copy(), 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, history

# ══════════════════════════════════════════════════
# 4. Evaluation & Metrics Calculation
# ══════════════════════════════════════════════════
def evaluate(model, X_te, y_te, device):
    """
    Complete evaluation to generate metrics for the dashboard.
    """
    model.eval()
    ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in dl:
            probs = torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(yb.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = y_prob.argmax(axis=1)

    acc = (y_pred == y_true).mean()
    return acc, y_pred, y_prob

# ══════════════════════════════════════════════════
# 5. Comprehensive Visualizations
# ✅ Removed Colab IPython.display dependency
# ══════════════════════════════════════════════════
def _save(fig, fname):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()  # 本地直接显示图表
    plt.close(fig)

def plot_class_distribution(y_tr, y_val, y_te):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Dataset Class Distribution", fontsize=16, fontweight="bold")
    sets = [("Train Set", y_tr), ("Val Set", y_val), ("Test Set", y_te)]

    for i, (title, data) in enumerate(sets):
        classes, counts = np.unique(data, return_counts=True)
        names = [CLASS_NAMES[c] for c in classes]
        bars = axes[i].bar(names, counts, color=COLORS, edgecolor="white")
        axes[i].set_title(title)
        axes[i].set_ylabel("Samples")
        axes[i].tick_params(axis='x', rotation=20)
        for bar in bars:
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{int(bar.get_height())}', ha='center', va='bottom')
    plt.tight_layout()
    _save(fig, "01_distribution.png")

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Learning Performance", fontsize=16, fontweight="bold")
    epochs = range(1, len(history["loss"]) + 1)

    ax1.plot(epochs, history["loss"], 'o-', label="Train Loss", ms=3)
    ax1.plot(epochs, history["val_loss"], 's-', label="Val Loss", ms=3)
    ax1.set_title("Loss Convergence"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["accuracy"], 'o-', label="Train Acc", ms=3)
    ax2.plot(epochs, history["val_accuracy"], 's-', label="Val Acc", ms=3)
    ax2.set_title("Accuracy Growth"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "02_history.png")

def plot_summary_dashboard(history, y_te, y_pred, y_prob, tflite_kb, n_params):
    """
    FULLY RESTORED: 6-Panel analytical dashboard.
    """
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Smart Posture Shirt · 1D-CNN Training Report", fontsize=20, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.3)
    epochs = range(1, len(history["accuracy"]) + 1)

    # 1. Accuracy Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["accuracy"], color=COLORS[0], label="Train")
    ax1.plot(epochs, history["val_accuracy"], color=COLORS[1], label="Val")
    ax1.set_title("Training Accuracy", fontsize=14); ax1.legend(); ax1.grid(alpha=0.2)

    # 2. Loss Curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history["loss"], color=COLORS[2], label="Train")
    ax2.plot(epochs, history["val_loss"], color=COLORS[3], label="Val")
    ax2.set_title("Training Loss", fontsize=14); ax2.legend(); ax2.grid(alpha=0.2)

    # 3. Normalized Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(y_te, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    im = ax3.imshow(cm_pct, cmap="Blues")
    ax3.set_title("Confusion Matrix (%)", fontsize=14)
    short_names = ["Upright", "Hunchback", "L.Side", "R.Side"]
    ax3.set_xticks(range(4)); ax3.set_xticklabels(short_names, rotation=15)
    ax3.set_yticks(range(4)); ax3.set_yticklabels(short_names)
    for i in range(4):
        for j in range(4):
            ax3.text(j, i, f"{cm_pct[i,j]:.0f}%", ha="center", va="center",
                     color="white" if cm_pct[i,j] > 50 else "black")

    # 4. F1 Scores
    ax4 = fig.add_subplot(gs[1, 0])
    f1s = f1_score(y_te, y_pred, average=None)
    ax4.bar(short_names, f1s, color=COLORS, edgecolor="white")
    ax4.set_title("F1 Score per Class", fontsize=14); ax4.set_ylim(0, 1.2)
    for i, v in enumerate(f1s):
        ax4.text(i, v + 0.02, f"{v:.3f}", ha="center")

    # 5. Confidence Histogram
    ax5 = fig.add_subplot(gs[1, 1])
    correct_mask = (y_pred == y_te)
    confidences = y_prob.max(axis=1)
    ax5.hist(confidences[correct_mask], bins=20, alpha=0.6, color=COLORS[2], label="Correct")
    ax5.hist(confidences[~correct_mask], bins=20, alpha=0.6, color=COLORS[3], label="Wrong")
    ax5.set_title("Prediction Confidence", fontsize=14); ax5.legend()

    # 6. Deployment Metrics Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    ax6.set_title("Deployment Metrics", fontweight="bold", fontsize=14)
    stats = [
        ("Test Accuracy", f"{(y_pred == y_te).mean():.2%}"),
        ("Best Val Acc", f"{max(history['val_accuracy']):.2%}"),
        ("Model Params", f"{n_params:,}"),
        ("TFLite Size", f"{tflite_kb:.1f} KB"),
        ("Quantization", "INT8 (Full)"),
        ("Sample Rate", f"{SAMPLE_RATE} Hz"),
        ("Target MCU", "ESP32-C6")
    ]
    for i, (k, v) in enumerate(stats):
        y_pos = 0.85 - i*0.13
        ax6.text(0.05, y_pos, k, transform=ax6.transAxes, fontsize=11, color="gray")
        ax6.text(0.95, y_pos, v, transform=ax6.transAxes, fontsize=12, fontweight="bold", ha="right")
        ax6.axhline(y_pos-0.04, xmin=0.05, xmax=0.95, color="#EEEEEE", lw=1)

    _save(fig, "03_dashboard.png")

# ══════════════════════════════════════════════════
# 6. Deployment: TFLite & C Header Export
# ══════════════════════════════════════════════════
def export_tflite_mcu(pt_model, X_tr, mean, std):
    """
    Generates INT8 quantized TFLite and C Header for ESP32.
    """
    print("\n[7/7] Starting TinyML Quantization Pipeline...")
    from tensorflow import keras
    from tensorflow.keras import layers as L

    # Map weights to TF/Keras
    tf_model = keras.Sequential([
        keras.Input(shape=(WIN, N_SENSORS)),
        L.Conv1D(16, 5, padding="same", activation="relu", name="c1"), L.MaxPooling1D(2),
        L.Conv1D(32, 3, padding="same", activation="relu", name="c2"), L.MaxPooling1D(2),
        L.Conv1D(64, 3, padding="same", activation="relu", name="c3"), L.GlobalAveragePooling1D(),
        L.Dense(32, activation="relu", name="f1"), L.Dropout(0.3), L.Dense(N_CLASSES, activation="softmax", name="out")
    ])

    sd = pt_model.state_dict()
    for l, w, b, t in [("c1","conv1.weight","conv1.bias",lambda x:x.transpose(2,1,0)),
                       ("c2","conv2.weight","conv2.bias",lambda x:x.transpose(2,1,0)),
                       ("c3","conv3.weight","conv3.bias",lambda x:x.transpose(2,1,0)),
                       ("f1","fc.weight","fc.bias",lambda x:x.T),
                       ("out","out.weight","out.bias",lambda x:x.T)]:
        tf_model.get_layer(l).set_weights([t(sd[w].cpu().numpy()), sd[b].cpu().numpy()])

    def rep_gen():
        for i in range(100):
            yield [X_tr[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = converter.inference_output_type = tf.int8

    tflite_bin = converter.convert()

    # Save Files
    with open(os.path.join(OUT_DIR, "posture_model.tflite"), "wb") as f:
        f.write(tflite_bin)
    h_path = os.path.join(OUT_DIR, "posture_model.h")
    with open(h_path, "w") as f:
        f.write(f'#pragma once\n#include <stdint.h>\n'
                f'alignas(8) const unsigned char posture_model_data[] = {{ {", ".join([f"0x{b:02x}" for b in tflite_bin])} }};\n'
                f'const unsigned int posture_model_len = {len(tflite_bin)};\n'
                f'static const float norm_mean[{N_SENSORS}] = {{ {", ".join([f"{v:.6f}f" for v in mean.flatten()])} }};\n'
                f'static const float norm_std[{N_SENSORS}] = {{ {", ".join([f"{v:.6f}f" for v in std.flatten()])} }};\n')

    kb = len(tflite_bin)/1024
    print(f"✅ Deployment file ready: {h_path} ({kb:.1f} KB)")
    return kb

# ══════════════════════════════════════════════════
# 7. Main Execution Flow
# ══════════════════════════════════════════════════
def main():
    # 创建本地输出文件夹
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting device: {device}")
    print(f"Dataset path: {os.path.abspath(CSV_FOLDER)}")
    print(f"Output path: {os.path.abspath(OUT_DIR)}")

    # Data Loading & Prep
    X_raw, y_raw = load_data(CSV_FOLDER)
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_raw, y_raw, test_size=0.3,
                                                      stratify=y_raw, random_state=42)

    X_tr, y_tr = sliding_window(X_tr_r, y_tr_r)
    X_te, y_te = sliding_window(X_te_r, y_te_r)
    X_v, X_t, y_v, y_t = train_test_split(X_te, y_te, test_size=0.5, stratify=y_te, random_state=42)

    # Normalization
    mean, std = X_tr.mean(axis=(0,1), keepdims=True), X_tr.std(axis=(0,1), keepdims=True) + 1e-8
    X_tr, X_v, X_t = (X_tr-mean)/std, (X_v-mean)/std, (X_t-mean)/std

    # Plot initial distribution
    plot_class_distribution(y_tr, y_v, y_t)

    # Training
    model, history = train_model(X_tr, y_tr, X_v, y_v, device)

    # Final Plots & Deployment
    plot_training_history(history)
    acc, y_pred, y_prob = evaluate(model, X_t, y_t, device)

    tflite_kb = export_tflite_mcu(model, X_tr, mean, std)
    plot_summary_dashboard(history, y_t, y_pred, y_prob, tflite_kb, sum(p.numel() for p in model.parameters()))

    print(f"\nFinal Test Results:")
    print(classification_report(y_t, y_pred, target_names=CLASS_NAMES))
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main()