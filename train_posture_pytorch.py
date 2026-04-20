"""
智能正姿衣 - 体态识别模型训练脚本 (1D-CNN · PyTorch → TFLite Micro)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
硬件平台:  ESP32-C6 (RISC-V, 160MHz, 512KB SRAM, 4MB Flash)
传感器:    3x FSR + BMI160 六轴IMU
输入通道:  9 (fsr_left, fsr_right, fsr_spine, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
采样率:    50 Hz | 窗口: 100点(2s) | 步长: 50点
训练框架:  PyTorch (GPU加速)
导出格式:  INT8 全量化 TFLite
目标大小:  ≤ 40 KB
标签定义:  0=挺直  1=驼背  2=左低右高  3=左高右低
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
改动记录 v2:
  - Conv 通道: 8/16/32  → 16/32/64  (提升模型容量，仍远低于 100KB 上限)
  - Dropout:   0.4      → 0.3       (修复 val_acc > train_acc 问题)
  - 初始 LR:   1e-3     → 5e-4      (抑制 val loss 中段抖动)
  - 类别权重:  新增 1/count          (改善 Upright F1 偏低问题)
  - Batch Size: 128 (不变)
  - Epochs: 50 → 60 (配合更大容量留足收敛时间)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
运行方式:
  source ~/workspace/tf_env/bin/activate
  python train_posture_pytorch.py
"""

# ── 无显示器环境：必须在 import pyplot 之前设置 ──────────────────
import os
if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

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

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# ══════════════════════════════════════════════════
# ★ 中文字体兼容
# ══════════════════════════════════════════════════
def _setup_chinese_font():
    candidates = [
        "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
        "Noto Sans CJK SC", "Heiti TC",
    ]
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return None

_FONT   = _setup_chinese_font()
_USE_CN = _FONT is not None

def _t(cn: str, en: str) -> str:
    return cn if _USE_CN else en

# ══════════════════════════════════════════════════
# ★ 全局参数
# ══════════════════════════════════════════════════
SAMPLE_RATE = 50
WINDOW_SEC  = 2
WIN         = SAMPLE_RATE * WINDOW_SEC   # 100
STRIDE      = WIN // 2                   # 50
N_SENSORS   = 9
N_CLASSES   = 4

CLASS_NAMES = [
    _t("挺直",    "Upright"),
    _t("驼背",    "Hunchback"),
    _t("左低右高", "LeftLow_RightHigh"),
    _t("左高右低", "LeftHigh_RightLow"),
]

SENSOR_COLS = [
    "fsr_left", "fsr_right", "fsr_spine",
    "acc_x", "acc_y", "acc_z",
    "gyr_x",  "gyr_y",  "gyr_z",
]
LABEL_COL = "label"
TIME_COL  = "time"

# ★ 修改为你的实际数据路径
CSV_FOLDER = os.path.expanduser("~/ai_study/posture_data/posture_synthetic_data")

OUT_DIR  = os.path.expanduser("~/ai_study/output")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
COLORS   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── 训练超参数 ────────────────────────────────────
EPOCHS      = 60       # v2: 50→60，给更大容量的模型留足收敛时间
BATCH_SIZE  = 128
LR          = 5e-4     # v2: 1e-3→5e-4
PATIENCE    = 12
LR_PATIENCE = 6
LR_FACTOR   = 0.5
LR_MIN      = 1e-6

# ══════════════════════════════════════════════════
# 1. 数据加载
# ══════════════════════════════════════════════════
def load_data(folder: str):
    files = glob.glob(os.path.join(folder, "**/*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"在 {folder} 中未找到 CSV 文件")

    dfs, skipped = [], 0
    for f in sorted(files):
        df = pd.read_csv(f)
        missing = [c for c in SENSOR_COLS + [LABEL_COL] if c not in df.columns]
        if missing:
            print(f"  ⚠️  跳过 {os.path.basename(f)}：缺列 {missing}")
            skipped += 1
            continue

        if TIME_COL in df.columns:
            dt = df[TIME_COL].diff().dropna()
            dt_mean = dt.mean()
            actual_hz = round(1.0 / dt_mean) if dt_mean > 0 else 0
            if abs(actual_hz - SAMPLE_RATE) > 2:
                print(f"  ⚠️  {os.path.basename(f)}: 采样率 {actual_hz}Hz ≠ {SAMPLE_RATE}Hz")
            gaps = dt[dt > 3.0 / SAMPLE_RATE]
            if len(gaps):
                print(f"  ⚠️  {os.path.basename(f)}: {len(gaps)} 处数据缺口")

        dfs.append(df[SENSOR_COLS + [LABEL_COL]])

    if not dfs:
        raise ValueError("没有合法的 CSV 文件")

    df_all  = pd.concat(dfs, ignore_index=True)
    raw_len = len(df_all)
    df_all  = df_all.replace([np.inf, -np.inf], np.nan).dropna()
    df_all  = df_all[df_all[LABEL_COL].isin(range(N_CLASSES))]
    cleaned = raw_len - len(df_all)
    if cleaned:
        print(f"  ⚠️  清除 {cleaned} 行无效数据")

    X = df_all[SENSOR_COLS].values.astype(np.float32)
    y = df_all[LABEL_COL].values.astype(np.int64)

    print(f"  加载 {len(files)-skipped} 个文件（跳过 {skipped}），共 {len(X):,} 行")
    dist = {CLASS_NAMES[k]: int(v)
            for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"  类别分布: {dist}")
    return X, y

# ══════════════════════════════════════════════════
# 2. 滑动窗口
# ══════════════════════════════════════════════════
def sliding_window(X: np.ndarray, y: np.ndarray,
                   win: int = WIN, stride: int = STRIDE):
    Xs, ys = [], []
    for i in range(0, len(X) - win, stride):
        Xs.append(X[i:i + win])
        ys.append(np.bincount(y[i:i + win]).argmax())
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)

# ══════════════════════════════════════════════════
# 3. 模型  v2: Conv 16/32/64，Dropout 0.3
#    输入: (batch, WIN=100, N_SENSORS=9)
# ══════════════════════════════════════════════════
class Posture1DCNN(nn.Module):
    def __init__(self, n_sensors=N_SENSORS, n_classes=N_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(n_sensors, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16,        32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32,        64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool1d(2)
        self.relu  = nn.ReLU()
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(64, 32)
        self.drop  = nn.Dropout(0.3)   # v2: 0.4 → 0.3
        self.out   = nn.Linear(32, n_classes)

    def forward(self, x):
        # (B, 100, 9) → (B, 9, 100)
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))   # → (B, 16, 50)
        x = self.pool(self.relu(self.conv2(x)))   # → (B, 32, 25)
        x = self.relu(self.conv3(x))              # → (B, 64, 25)
        x = self.gap(x).squeeze(-1)               # → (B, 64)
        x = self.relu(self.fc(x))                 # → (B, 32)
        x = self.drop(x)
        return self.out(x)                        # → (B, 4)  logits

# ══════════════════════════════════════════════════
# 4. 训练循环
# ══════════════════════════════════════════════════
def train_model(X_tr, y_tr, X_val, y_val, device):
    model    = Posture1DCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数量: {n_params:,}")

    tr_ds  = TensorDataset(torch.from_numpy(X_tr),  torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # v2: 类别权重 —— 补偿 Upright 样本最多导致的偏置
    counts   = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float32)
    weights  = 1.0 / counts
    weights  = weights / weights.sum() * N_CLASSES   # 归一化，均值=1
    w_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"  类别权重: { {CLASS_NAMES[i]: f'{weights[i]:.3f}' for i in range(N_CLASSES)} }")

    criterion = nn.CrossEntropyLoss(weight=w_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_PATIENCE, factor=LR_FACTOR,
        min_lr=LR_MIN)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0

    for epoch in range(1, EPOCHS + 1):
        # ── 训练 ──────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * len(yb)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += len(yb)

        # ── 验证 ──────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                val_loss    += criterion(logits, yb).item() * len(yb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total   += len(yb)

        ep_loss     = tr_loss    / tr_total
        ep_acc      = tr_correct / tr_total
        ep_val_loss = val_loss   / val_total
        ep_val_acc  = val_correct / val_total

        history["loss"].append(ep_loss)
        history["accuracy"].append(ep_acc)
        history["val_loss"].append(ep_val_loss)
        history["val_accuracy"].append(ep_val_acc)

        scheduler.step(ep_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:>3}/{EPOCHS}  "
              f"loss={ep_loss:.4f}  acc={ep_acc:.4f}  "
              f"val_loss={ep_val_loss:.4f}  val_acc={ep_val_acc:.4f}  "
              f"lr={current_lr:.2e}")

        if ep_val_acc > best_val_acc:
            best_val_acc = ep_val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  ⏹  EarlyStopping at epoch {epoch}，最佳 val_acc={best_val_acc:.4f}")
                break

    model.load_state_dict(best_state)
    return model, history, n_params

# ══════════════════════════════════════════════════
# 5. 评估
# ══════════════════════════════════════════════════
def evaluate(model, X_te, y_te, device):
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
    return (y_pred == y_true).mean(), y_pred, y_prob

# ══════════════════════════════════════════════════
# 6. PyTorch → TF Keras 权重迁移 → INT8 TFLite
# ══════════════════════════════════════════════════
def export_tflite(pt_model, X_tr, mean_flat, std_flat):
    print("  导入 TensorFlow 进行 TFLite 转换（强制 CPU 模式）...")

    # ★ 必须在 import tensorflow 之前设置，否则 TF 会尝试初始化 GPU
    # RTX 5060 + CUDA 13 与当前 TF nightly 存在兼容性问题，
    # 转换步骤只需 CPU，训练已由 PyTorch GPU 完成
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers as L
    except Exception as e:
        print(f"  ❌ TensorFlow 导入失败: {e}")
        print("     请确认已激活 gpu_env: source ~/workspace/tf_env/bin/activate")
        raise

    # v2: 通道与 PyTorch 保持一致 (16/32/64)
    tf_model = keras.Sequential([
        keras.Input(shape=(WIN, N_SENSORS), name="input"),
        L.Conv1D(16, 5, padding="same", activation="relu", name="conv1"),
        L.MaxPooling1D(2, name="pool1"),
        L.Conv1D(32, 3, padding="same", activation="relu", name="conv2"),
        L.MaxPooling1D(2, name="pool2"),
        L.Conv1D(64, 3, padding="same", activation="relu", name="conv3"),
        L.GlobalAveragePooling1D(name="gap"),
        L.Dense(32, activation="relu", name="fc"),
        L.Dropout(0.3, name="dropout"),
        L.Dense(N_CLASSES, activation="softmax", name="output"),
    ], name="posture_1dcnn")
    tf_model.build(input_shape=(None, WIN, N_SENSORS))

    # 权重迁移
    # Conv1d  PT: (out_ch, in_ch, k)  → TF Conv1D: (k, in_ch, out_ch)
    # Linear  PT: (out, in)           → TF Dense:  (in, out)
    sd = pt_model.state_dict()
    for layer_name, pt_w_key, pt_b_key, w_transform in [
        ("conv1", "conv1.weight", "conv1.bias", lambda w: w.numpy().transpose(2, 1, 0)),
        ("conv2", "conv2.weight", "conv2.bias", lambda w: w.numpy().transpose(2, 1, 0)),
        ("conv3", "conv3.weight", "conv3.bias", lambda w: w.numpy().transpose(2, 1, 0)),
        ("fc",    "fc.weight",    "fc.bias",    lambda w: w.numpy().T),
        ("output","out.weight",   "out.bias",   lambda w: w.numpy().T),
    ]:
        tf_model.get_layer(layer_name).set_weights([
            w_transform(sd[pt_w_key].cpu()),
            sd[pt_b_key].cpu().numpy(),
        ])
    print("  ✅ 权重迁移完成")

    # ── INT8 全量化 ───────────────────────────────────────────
    print("  开始 INT8 全量化转换...")
    n_rep       = min(500, len(X_tr))
    rep_indices = np.random.choice(len(X_tr), n_rep, replace=False)

    def rep_data():
        for idx in rep_indices:
            yield [X_tr[idx:idx + 1].astype(np.float32)]

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        converter.optimizations              = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset     = rep_data
        converter.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type       = tf.int8
        converter.inference_output_type      = tf.int8
        tflite = converter.convert()
    except Exception as e:
        print(f"  ❌ TFLite 转换失败: {e}")
        print("     请把完整报错发给我排查")
        raise

    tflite_kb   = len(tflite) / 1024
    tflite_path = os.path.join(OUT_DIR, "posture_model.tflite")
    header_path = os.path.join(OUT_DIR, "posture_model.h")

    with open(tflite_path, "wb") as f:
        f.write(tflite)
    print(f"  ✅ posture_model.tflite -> {tflite_path}")

    with open(header_path, "w", encoding="utf-8") as f:
        f.write(_tflite_to_c_header(tflite, mean_flat, std_flat))
    print(f"  ✅ posture_model.h      -> {header_path}")

    print(f"  TFLite 大小: {tflite_kb:.1f} KB  "
          f"({'✅ <=40KB 达标' if tflite_kb <= 40 else '❌ 超限，考虑减少 conv3 通道'})")
    return tflite_kb

def _tflite_to_c_header(tflite_bytes, mean_flat, std_flat):
    kb = len(tflite_bytes) / 1024
    hex_lines = []
    for i in range(0, len(tflite_bytes), 16):
        chunk = tflite_bytes[i:i + 16]
        hex_lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk))
    hex_body  = ",\n".join(hex_lines)
    mean_str  = ", ".join(f"{v:.8f}f" for v in mean_flat)
    std_str   = ", ".join(f"{v:.8f}f" for v in std_flat)

    return f"""// ============================================================
// 智能正姿衣 · 体态识别模型  v2
// 自动生成，勿手动修改
// ------------------------------------------------------------
// 模型大小  : {kb:.1f} KB (INT8 全量化)
// 目标芯片  : ESP32-C6 (RISC-V, TFLite Micro)
// 输入形状  : [{WIN}, {N_SENSORS}]  (100点 x 9通道)
// 输出形状  : [{N_CLASSES}]  (softmax 0=挺直 1=驼背 2=左低右高 3=左高右低)
// Conv 通道 : 16 / 32 / 64
// 采样率    : {SAMPLE_RATE} Hz，窗口 {WINDOW_SEC}s
// ============================================================
#pragma once
#include <stdint.h>

alignas(8) static const uint8_t posture_model_tflite[] = {{
{hex_body}
}};
static const unsigned int posture_model_tflite_len = {len(tflite_bytes)}U;

// ── 归一化参数（推理前必须使用）──────────────────────────────
// 通道顺序: {", ".join(SENSOR_COLS)}
// ESP32 预处理: input_norm[i] = (raw[i] - norm_mean[i]) / norm_std[i]
static const float norm_mean[{N_SENSORS}] = {{{mean_str}}};
static const float norm_std[{N_SENSORS}]  = {{{std_str}}};
"""

# ══════════════════════════════════════════════════
# 7. 可视化
# ══════════════════════════════════════════════════
def _save(fig, fname):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 {path}")

def plot_class_distribution(y_tr, y_val, y_te):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(_t("各数据集类别分布", "Class Distribution"),
                 fontsize=13, fontweight="bold")
    for ax, y, title in zip(axes, [y_tr, y_val, y_te],
                             [_t("训练集","Train"), _t("验证集","Val"), _t("测试集","Test")]):
        classes, counts = np.unique(y, return_counts=True)
        bars = ax.bar([CLASS_NAMES[c] for c in classes],
                      counts, color=COLORS[:len(classes)], edgecolor="white")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    str(cnt), ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(_t("样本数", "Count"))
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "01_class_distribution.png")

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(_t("模型训练过程","Training History"), fontsize=13, fontweight="bold")
    ep = range(1, len(history["loss"]) + 1)

    ax1.plot(ep, history["loss"],     "o-", color=COLORS[0], ms=3, label=_t("训练","Train"))
    ax1.plot(ep, history["val_loss"], "s-", color=COLORS[1], ms=3, label=_t("验证","Val"))
    ax1.set_title(_t("损失曲线","Loss")); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(alpha=0.3); ax1.spines[["top","right"]].set_visible(False)

    best_ep  = int(np.argmax(history["val_accuracy"])) + 1
    best_acc = max(history["val_accuracy"])
    ax2.plot(ep, history["accuracy"],     "o-", color=COLORS[2], ms=3, label=_t("训练","Train"))
    ax2.plot(ep, history["val_accuracy"], "s-", color=COLORS[3], ms=3, label=_t("验证","Val"))
    ax2.axvline(best_ep, color="gray", linestyle="--", alpha=0.6)
    ax2.annotate(f'{_t("最佳","Best")} Ep{best_ep}\n{best_acc:.3f}',
                 xy=(best_ep, best_acc), xytext=(best_ep+1, best_acc-0.05),
                 arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)
    ax2.set_ylim(0, 1.05); ax2.set_xlabel("Epoch")
    ax2.set_title(_t("准确率曲线","Accuracy")); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "02_training_history.png")

def plot_confusion_matrix(y_true, y_pred):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(_t("混淆矩阵","Confusion Matrix"), fontsize=13, fontweight="bold")
    for ax, data, fmt, sub in zip(
            axes, [cm, cm_pct], ["d", ".1f"],
            [_t("原始数量","Count"), _t("行归一化 (%)","Row-norm (%)")]):
        im = ax.imshow(data, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel(_t("预测标签","Predicted"))
        ax.set_ylabel(_t("真实标签","True"))
        ax.set_title(sub)
        thresh = data.max() / 2
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                sfx = "%" if fmt == ".1f" else ""
                ax.text(j, i, f"{data[i,j]:{fmt}}{sfx}",
                        ha="center", va="center", fontsize=10,
                        color="white" if data[i,j] > thresh else "black")
    plt.tight_layout()
    _save(fig, "03_confusion_matrix.png")

def plot_confidence_distribution(y_true, y_prob):
    fig, axes = plt.subplots(1, N_CLASSES, figsize=(4*N_CLASSES, 4), sharey=True)
    fig.suptitle(_t("各类别预测置信度","Confidence Distribution"),
                 fontsize=13, fontweight="bold")
    for i, (ax, name) in enumerate(zip(axes, CLASS_NAMES)):
        mask = y_true == i
        if not mask.any():
            ax.set_title(f"{name}\n(no sample)"); continue
        ax.hist(y_prob[mask, i], bins=20, range=(0, 1),
                color=COLORS[i], edgecolor="white", alpha=0.85)
        ax.axvline(y_prob[mask, i].mean(), color="black", linestyle="--",
                   label=f'mean={y_prob[mask,i].mean():.2f}')
        ax.set_title(f"{name} (n={mask.sum()})")
        ax.set_xlabel(_t("置信度","Confidence"))
        ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel(_t("样本数","Count"))
    plt.tight_layout()
    _save(fig, "04_confidence_dist.png")

def plot_summary_dashboard(history, y_te, y_pred, y_prob, tflite_kb, n_params):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(_t("智能正姿衣 · 体态识别模型训练报告",
                    "Smart Posture Shirt · Training Report"),
                 fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ep = range(1, len(history["accuracy"]) + 1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep, history["accuracy"],     color=COLORS[0], label=_t("训练","Train"))
    ax1.plot(ep, history["val_accuracy"], color=COLORS[1], label=_t("验证","Val"))
    ax1.set_title(_t("准确率","Accuracy")); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(alpha=0.3); ax1.spines[["top","right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ep, history["loss"],     color=COLORS[2], label=_t("训练","Train"))
    ax2.plot(ep, history["val_loss"], color=COLORS[3], label=_t("验证","Val"))
    ax2.set_title(_t("损失","Loss")); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(alpha=0.3); ax2.spines[["top","right"]].set_visible(False)

    ax3 = fig.add_subplot(gs[0, 2])
    cm_pct = confusion_matrix(y_te, y_pred).astype(float)
    cm_pct = cm_pct / cm_pct.sum(axis=1, keepdims=True) * 100
    ax3.imshow(cm_pct, cmap="Blues")
    ax3.set_xticks(range(N_CLASSES)); ax3.set_yticks(range(N_CLASSES))
    ax3.set_xticklabels(CLASS_NAMES, fontsize=7)
    ax3.set_yticklabels(CLASS_NAMES, fontsize=7)
    ax3.set_title(_t("混淆矩阵 (%)","Confusion (%)"))
    thresh = cm_pct.max() / 2
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax3.text(j, i, f"{cm_pct[i,j]:.0f}%", ha="center", va="center",
                     fontsize=9, color="white" if cm_pct[i,j] > thresh else "black")

    ax4 = fig.add_subplot(gs[1, 0])
    f1s  = f1_score(y_te, y_pred, average=None)
    bars = ax4.bar(CLASS_NAMES, f1s, color=COLORS, edgecolor="white")
    ax4.set_ylim(0, 1.15); ax4.set_title("F1 Score per Class")
    for bar, v in zip(bars, f1s):
        ax4.text(bar.get_x()+bar.get_width()/2, v+0.02,
                 f"{v:.3f}", ha="center", fontsize=9)
    ax4.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="0.9 baseline")
    ax4.legend(fontsize=8); ax4.spines[["top","right"]].set_visible(False)

    ax5 = fig.add_subplot(gs[1, 1])
    ok   = y_pred == y_te
    conf = y_prob.max(axis=1)
    ax5.hist(conf[ok],  bins=20, alpha=0.75, color=COLORS[2],
             label=f'{_t("正确","Correct")} n={ok.sum()}')
    ax5.hist(conf[~ok], bins=20, alpha=0.75, color=COLORS[3],
             label=f'{_t("错误","Wrong")} n={(~ok).sum()}')
    ax5.set_title(_t("最大置信度分布","Max Confidence"))
    ax5.set_xlabel(_t("置信度","Confidence"))
    ax5.legend(); ax5.spines[["top","right"]].set_visible(False)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    ax6.set_title(_t("模型指标","Model Metrics"), fontweight="bold", pad=10)
    size_ok = tflite_kb <= 40
    metrics = [
        (_t("测试准确率",     "Test Acc"),     f"{ok.mean():.2%}"),
        (_t("最佳验证准确率", "Best Val Acc"), f"{max(history['val_accuracy']):.2%}"),
        (_t("训练 Epochs",    "Epochs"),       str(len(ep))),
        (_t("模型参数量",     "Params"),       f"{n_params:,}"),
        ("TFLite (INT8)",                       f"{tflite_kb:.1f} KB  {'✅' if size_ok else '❌'}"),
        (_t("Conv 通道",      "Conv CH"),      "16 / 32 / 64"),
        (_t("目标芯片",       "MCU"),          "ESP32-C6"),
    ]
    for idx, (k, v) in enumerate(metrics):
        yp = 0.90 - idx * 0.11
        ax6.text(0.05, yp, k, transform=ax6.transAxes, fontsize=9.5, color="gray")
        ax6.text(0.95, yp, v, transform=ax6.transAxes, fontsize=10,
                 fontweight="bold", ha="right",
                 color=COLORS[0] if idx == 0 else "black")
        ax6.axhline(yp-0.035, xmin=0.05, xmax=0.95, color="#eeeeee", linewidth=0.8)
    _save(fig, "05_summary_dashboard.png")

# ══════════════════════════════════════════════════
# 8. 主流程
# ══════════════════════════════════════════════════
def main(csv_folder: str = CSV_FOLDER):
    os.makedirs(OUT_DIR,  exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("智能正姿衣 · 1D-CNN 体态识别  v2  (PyTorch)")
    print(f"  设备: {device}"
          + (f"  [{torch.cuda.get_device_name(0)}]"
             if torch.cuda.is_available() else ""))
    print(f"  Conv: 16/32/64 | Dropout: 0.3 | LR: {LR} | BS: {BATCH_SIZE}")
    print(f"  数据: {csv_folder}")
    print("=" * 60)

    # [1/5] 加载
    print("\n[1/5] 加载数据...")
    X_raw, y_raw = load_data(csv_folder)

    # [2/5] 分割 + 滑窗
    print("\n[2/5] 数据分割与滑动窗口...")
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_raw, y_raw, test_size=0.3, stratify=y_raw, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    X_tr,  y_tr  = sliding_window(X_tr,  y_tr)
    X_val, y_val = sliding_window(X_val, y_val)
    X_te,  y_te  = sliding_window(X_te,  y_te)
    print(f"  训练: {X_tr.shape} | 验证: {X_val.shape} | 测试: {X_te.shape}")

    plot_class_distribution(y_tr, y_val, y_te)

    # 归一化（仅用训练集统计）
    mean = X_tr.mean(axis=(0, 1), keepdims=True)
    std  = X_tr.std(axis=(0, 1),  keepdims=True) + 1e-8
    X_tr  = (X_tr  - mean) / std
    X_val = (X_val - mean) / std
    X_te  = (X_te  - mean) / std

    mean_flat = mean.flatten().astype(np.float32)
    std_flat  = std.flatten().astype(np.float32)
    np.save(os.path.join(OUT_DIR, "norm_mean_flat.npy"), mean_flat)
    np.save(os.path.join(OUT_DIR, "norm_std_flat.npy"),  std_flat)

    # [3/5] 训练
    print("\n[3/5] 构建并训练模型...")
    t0 = time.time()
    model, history, n_params = train_model(X_tr, y_tr, X_val, y_val, device)
    print(f"  训练耗时: {time.time()-t0:.1f}s")
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "posture_model.pt"))

    # [4/5] 评估
    print("\n[4/5] 模型评估与可视化...")
    plot_training_history(history)
    acc, y_pred, y_prob = evaluate(model, X_te, y_te, device)
    print(f"\n  测试准确率: {acc:.4f}")
    print("\n" + classification_report(
        y_te, y_pred, target_names=CLASS_NAMES, digits=4))
    plot_confusion_matrix(y_te, y_pred)
    plot_confidence_distribution(y_te, y_prob)

    # [5/5] TFLite 导出
    print("\n[5/5] 导出 TFLite (INT8 全量化)...")
    tflite_kb = export_tflite(model, X_tr, mean_flat, std_flat)

    plot_summary_dashboard(history, y_te, y_pred, y_prob, tflite_kb, n_params)

    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print(f"  测试准确率:  {acc:.2%}")
    print(f"  TFLite 大小: {tflite_kb:.1f} KB  "
          f"({'✅ ≤40KB 达标' if tflite_kb <= 40 else '❌ 超限'})")
    print(f"  参数量:      {n_params:,}")
    print(f"\n  输出目录: {OUT_DIR}/")
    print("    posture_model.tflite  ← 烧录到 ESP32-C6")
    print("    posture_model.h       ← 复制到 ESP32 工程 ★")
    print("    posture_model.pt      ← PyTorch 权重（微调用）")
    print("    plots/                ← 可视化图表 (5张)")
    print("=" * 60)

if __name__ == "__main__":
    main(csv_folder=CSV_FOLDER)