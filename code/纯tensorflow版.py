"""
智能正姿衣 - 体态识别模型训练脚本 (1D-CNN · TFLite Micro)  [TensorFlow 纯净版]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
硬件平台:  ESP32-C6 (RISC-V, 160MHz, 512KB SRAM, 4MB Flash)
传感器:    3x FSR + BMI160 六轴IMU
输入通道:  9 (fsr_left, fsr_right, fsr_spine, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
采样率:    50 Hz (定时器同步采样)
窗口大小:  100 点 = 2 秒
滑动步长:  50  点 = 1 秒 (50% overlap)
量化方式:  INT8 全量化
目标大小:  ≤ 100 KB (编译进 Flash，运行时加载入 SRAM)
标签定义:  0=挺直  1=驼背  2=左低右高  3=左高右低
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
运行说明：
1. 确保 GPU 可用（需 CUDA + cuDNN + TF-GPU 版本匹配）
   - RTX 50系 (Blackwell) 目前 TF 不支持 GPU，自动回退 CPU
2. 数据集路径：默认 ~/ai_study/posture_data/posture_synthetic_data/
3. 安装依赖：~/ai_study/gpu_env/bin/pip install tensorflow pandas numpy scikit-learn matplotlib
4. 运行：~/ai_study/gpu_env/bin/python -u ~/ai_study/train_posture_tensorflow.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

# ══════════════════════════════════════════════════
# ★ 中文字体兼容（多平台）
# ══════════════════════════════════════════════════
def _setup_chinese_font():
    candidates = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
        'Heiti TC', 'Noto Sans CJK SC'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return None

_FONT = _setup_chinese_font()
_USE_CN = _FONT is not None

def _t(cn: str, en: str) -> str:
    return cn if _USE_CN else en

# ══════════════════════════════════════════════════
# ★ GPU 检测 & TensorFlow 延迟导入
# ══════════════════════════════════════════════════
# 先检测 GPU，再导入 TensorFlow
# 这样可以在导入前设置环境变量（如需要强制 CPU）

def _detect_gpu():
    """检测系统 GPU 是否可用于 TensorFlow"""
    import subprocess
    gpu_info = {"available": False, "name": "N/A", "cuda_version": "N/A", "compute_cap": "N/A"}

    # 1. 检查 nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            gpu_info["name"] = parts[0]
            gpu_info["compute_cap"] = parts[1] if len(parts) > 1 else "N/A"
    except Exception:
        return gpu_info

    # 2. 检查 CUDA 版本
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=cuda_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info["cuda_version"] = result.stdout.strip()
    except Exception:
        pass

    return gpu_info

_gpu_info = _detect_gpu()


_FORCE_CPU = False
if _gpu_info["compute_cap"].startswith("12."):
    _FORCE_CPU = True
    print(f"⚠️  检测到 GPU: {_gpu_info['name']} (Compute Capability {_gpu_info['compute_cap']})")
    print("⚠️  Blackwell 架构目前 TensorFlow 不支持 GPU，将强制使用 CPU")
    print("⚠️  如果未来 TF 更新支持，可删除脚本中 _FORCE_CPU=True 的逻辑\n")

if _FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 现在才导入 TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 打印 TF 信息
_tf_gpu = len(tf.config.list_physical_devices('GPU')) > 0
print(f"TensorFlow 版本: {tf.__version__}")
print(f"GPU 可用: {_tf_gpu}")
if _tf_gpu:
    for dev in tf.config.list_physical_devices('GPU'):
        print(f"  → {dev.name}")
else:
    print("  → 使用 CPU 模式训练（速度较慢）")
print()

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# ══════════════════════════════════════════════════
# ★ 全局参数
# ══════════════════════════════════════════════════
SAMPLE_RATE = 50
WINDOW_SEC  = 2
WIN         = SAMPLE_RATE * WINDOW_SEC   # 100 点
STRIDE      = WIN // 2                   # 50  点
N_SENSORS   = 9
N_CLASSES   = 4

CLASS_NAMES = [
    _t('挺直', 'Upright'),
    _t('驼背', 'Hunchback'),
    _t('左低右高', 'LeftLow_RightHigh'),
    _t('左高右低', 'LeftHigh_RightLow'),
]

SENSOR_COLS = [
    'fsr_left', 'fsr_right', 'fsr_spine',
    'acc_x', 'acc_y', 'acc_z',
    'gyr_x',  'gyr_y',  'gyr_z',
]
LABEL_COL = 'label'
TIME_COL  = 'time'

# ★ 路径配置（适配 WSL 环境）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FOLDER = os.path.join(SCRIPT_DIR, "posture_data", "posture_synthetic_data")
OUT_DIR    = os.path.join(SCRIPT_DIR, "output_tf")
PLOT_DIR   = os.path.join(OUT_DIR, "plots")
COLORS     = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

# ══════════════════════════════════════════════════
# 1. 数据加载
# ══════════════════════════════════════════════════
def load_data(folder: str):
    """加载 CSV 数据，支持递归穿透子文件夹"""
    files = glob.glob(os.path.join(folder, "**/*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"在 {folder} 中未找到 CSV 文件，请检查数据集路径是否正确")

    print(f"  找到 {len(files)} 个 CSV 文件，开始加载...")

    dfs = []
    skipped = 0
    for f in sorted(files):
        df = pd.read_csv(f)
        missing = [c for c in SENSOR_COLS + [LABEL_COL] if c not in df.columns]
        if missing:
            print(f"  ⚠️  跳过 {os.path.basename(f)}：缺列 {missing}")
            skipped += 1
            continue

        # 时间戳采样率校验
        if TIME_COL in df.columns:
            dt        = df[TIME_COL].diff().dropna()
            dt_mean   = dt.mean()
            dt_std    = dt.std()
            actual_hz = round(1.0 / dt_mean) if dt_mean > 0 else 0

            if abs(actual_hz - SAMPLE_RATE) > 2:
                print(f"  ⚠️  {os.path.basename(f)}: 实测采样率 {actual_hz}Hz 偏离目标 {SAMPLE_RATE}Hz")
            if dt_std > 0.005:
                print(f"  ⚠️  {os.path.basename(f)}: 采样间隔抖动 std={dt_std:.4f}s")

            gaps = dt[dt > 3.0 / SAMPLE_RATE]
            if len(gaps):
                print(f"  ⚠️  {os.path.basename(f)}: 检测到 {len(gaps)} 处数据缺口")

            dur = df[TIME_COL].iloc[-1] - df[TIME_COL].iloc[0]
            print(f"  ✅  {os.path.basename(f)}: {actual_hz}Hz | {dur:.1f}s | {len(df)} 行")

        dfs.append(df[SENSOR_COLS + [LABEL_COL]])

    if not dfs:
        raise ValueError("没有合法的 CSV 文件可供加载，请检查数据集格式")

    df_all  = pd.concat(dfs, ignore_index=True)
    raw_len = len(df_all)

    # 基础数据校验
    df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna()
    df_all = df_all[df_all[LABEL_COL].isin([0, 1, 2, 3])]

    cleaned = raw_len - len(df_all)
    if cleaned > 0:
        print(f"  ⚠️  校验：已删除 {cleaned} 行无效数据（NaN/Inf/非法标签）")

    X = df_all[SENSOR_COLS].values.astype(np.float32)
    y = df_all[LABEL_COL].values.astype(np.int32)

    print(f"  加载 {len(files) - skipped} 个文件（跳过 {skipped} 个），共 {len(X)} 行有效数据")
    print(f"  类别分布: { {CLASS_NAMES[k]: int(v) for k, v in zip(*np.unique(y, return_counts=True))} }")
    return X, y

# ══════════════════════════════════════════════════
# 2. 滑动窗口切片
# ══════════════════════════════════════════════════
def sliding_window(X: np.ndarray, y: np.ndarray, win: int = WIN, stride: int = STRIDE):
    """将连续数据按滑动窗口切割成片段，标签取窗口内众数"""
    Xs, ys = [], []
    for i in range(0, len(X) - win, stride):
        Xs.append(X[i:i + win])
        ys.append(np.bincount(y[i:i + win]).argmax())
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)

# ══════════════════════════════════════════════════
# 3. 模型定义
# ══════════════════════════════════════════════════
def build_model():
    """构建 1D-CNN 体态识别模型"""
    return keras.Sequential([
        keras.Input(shape=(WIN, N_SENSORS), name='input'),
        layers.Conv1D(16,  5, padding='same', activation='relu', name='conv1'),
        layers.MaxPooling1D(2, name='pool1'),
        layers.Conv1D(32, 3, padding='same', activation='relu', name='conv2'),
        layers.MaxPooling1D(2, name='pool2'),
        layers.Conv1D(64, 3, padding='same', activation='relu', name='conv3'),
        layers.GlobalAveragePooling1D(name='gap'),
        layers.Dense(32, activation='relu', name='fc'),
        layers.Dropout(0.3, name='dropout'),
        layers.Dense(N_CLASSES, activation='softmax', name='output'),
    ], name='posture_1dcnn_tf')

# ══════════════════════════════════════════════════
# 4. TFLite → C 头文件
# ══════════════════════════════════════════════════
def tflite_to_c_header(tflite_bytes: bytes, var_name: str = "posture_model",
                       mean_flat: np.ndarray = None, std_flat: np.ndarray = None) -> str:
    """将 TFLite 字节码转换为 C 头文件，供 ESP32 编译使用"""
    kb = len(tflite_bytes) / 1024
    hex_lines = []
    for i in range(0, len(tflite_bytes), 16):
        chunk = tflite_bytes[i:i + 16]
        hex_lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk))
    hex_body = ",\n".join(hex_lines)

    norm_section = ""
    if mean_flat is not None and std_flat is not None:
        mean_str   = ", ".join(f"{v:.8f}f" for v in mean_flat)
        std_str    = ", ".join(f"{v:.8f}f" for v in std_flat)
        ch_comment = ", ".join(SENSOR_COLS)
        norm_section = f"""
// ── 归一化参数（推理前必须使用）──────────────
// 通道顺序: {ch_comment}
// ESP32 推理预处理: input_norm[i] = (raw[i] - norm_mean[i]) / norm_std[i]
static const float norm_mean[{N_SENSORS}] = {{{mean_str}}};
static const float norm_std[{N_SENSORS}]  = {{{std_str}}};
"""

    return f"""// ============================================================
// 智能正姿衣 · 体态识别模型 [TensorFlow 版]
// 自动生成，勿手动修改
// ------------------------------------------------------------
// 模型大小  : {kb:.1f} KB (INT8 全量化)
// 目标芯片  : ESP32-C6 (RISC-V, TFLite Micro)
// 输入形状  : [{WIN}, {N_SENSORS}]  (100点 x 9通道)
// 输出形状  : [{N_CLASSES}]  (softmax, 0=挺直 1=驼背 2=左低右高 3=左高右低)
// 采样率    : {SAMPLE_RATE} Hz，窗口 {WINDOW_SEC}s
// 训练框架  : TensorFlow {tf.__version__}
// ============================================================
#pragma once
#include <stdint.h>

alignas(8) static const uint8_t posture_model_tflite[] = {{
{hex_body}
}};
static const unsigned int posture_model_tflite_len = {len(tflite_bytes)}U;
{norm_section}"""

# ══════════════════════════════════════════════════
# 5. 可视化函数
# ══════════════════════════════════════════════════
def _savefig(fig, path):
    """保存图片（不弹出窗口，适合 WSL / 服务器环境）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_class_distribution(y_tr, y_val, y_te):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(_t('各数据集类别分布', 'Class Distribution'), fontsize=13, fontweight='bold')
    titles = [_t('训练集', 'Train'), _t('验证集', 'Val'), _t('测试集', 'Test')]
    for ax, y, title in zip(axes, [y_tr, y_val, y_te], titles):
        classes, counts = np.unique(y, return_counts=True)
        names = [CLASS_NAMES[c] for c in classes]
        bars = ax.bar(names, counts, color=COLORS[:len(classes)], edgecolor='white')
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(cnt),
                    ha='center', va='bottom', fontsize=10)
        ax.set_title(title)
        ax.set_ylabel(_t('样本数', 'Count'))
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _savefig(fig, os.path.join(PLOT_DIR, '01_class_distribution.png'))

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(_t('模型训练过程', 'Training History'), fontsize=13, fontweight='bold')
    ep = range(1, len(history.history['loss']) + 1)

    ax1.plot(ep, history.history['loss'],     'o-', color=COLORS[0], ms=3, label=_t('训练', 'Train'))
    ax1.plot(ep, history.history['val_loss'], 's-', color=COLORS[1], ms=3, label=_t('验证', 'Val'))
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title(_t('损失曲线', 'Loss')); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.spines[['top', 'right']].set_visible(False)

    best_ep  = np.argmax(history.history['val_accuracy']) + 1
    best_acc = max(history.history['val_accuracy'])
    ax2.plot(ep, history.history['accuracy'],     'o-', color=COLORS[2], ms=3, label=_t('训练', 'Train'))
    ax2.plot(ep, history.history['val_accuracy'], 's-', color=COLORS[3], ms=3, label=_t('验证', 'Val'))
    ax2.axvline(best_ep, color='gray', linestyle='--', alpha=0.6)
    ax2.annotate(f'{_t("最佳", "Best")} Ep{best_ep}\n{best_acc:.3f}',
                 xy=(best_ep, best_acc), xytext=(best_ep + 1, best_acc - 0.05),
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)
    ax2.set_ylim(0, 1.05); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title(_t('准确率曲线', 'Accuracy')); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    _savefig(fig, os.path.join(PLOT_DIR, '02_training_history.png'))

def plot_confusion_matrix(y_true, y_pred):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    n      = len(CLASS_NAMES)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(_t('混淆矩阵', 'Confusion Matrix'), fontsize=13, fontweight='bold')
    subtitles = [_t('原始数量', 'Count'), _t('行归一化 (%)', 'Row-norm (%)')]
    for ax, data, fmt, sub in zip(axes, [cm, cm_pct], ['d', '.1f'], subtitles):
        im = ax.imshow(data, cmap='Blues')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel(_t('预测标签', 'Predicted')); ax.set_ylabel(_t('真实标签', 'True'))
        ax.set_title(sub)
        thresh = data.max() / 2
        for i in range(n):
            for j in range(n):
                sfx = '%' if fmt == '.1f' else ''
                ax.text(j, i, f'{data[i, j]:{fmt}}{sfx}',
                        ha='center', va='center', fontsize=11,
                        color='white' if data[i, j] > thresh else 'black')
    plt.tight_layout()
    _savefig(fig, os.path.join(PLOT_DIR, '03_confusion_matrix.png'))

def plot_confidence_distribution(y_true, y_prob):
    fig, axes = plt.subplots(1, N_CLASSES, figsize=(4 * N_CLASSES, 4), sharey=True)
    fig.suptitle(_t('各类别预测置信度分布', 'Confidence Distribution'), fontsize=13, fontweight='bold')
    for i, (ax, name) in enumerate(zip(axes, CLASS_NAMES)):
        mask = y_true == i
        if not mask.any():
            ax.set_title(f'{name}\n({_t("无样本", "no sample")})'); continue
        ax.hist(y_prob[mask, i], bins=20, range=(0, 1),
                color=COLORS[i], edgecolor='white', alpha=0.85)
        ax.axvline(y_prob[mask, i].mean(), color='black', linestyle='--',
                   label=f'{_t("均值", "mean")}={y_prob[mask, i].mean():.2f}')
        ax.set_title(f'{name} (n={mask.sum()})')
        ax.set_xlabel(_t('置信度', 'Confidence'))
        ax.legend(fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel(_t('样本数', 'Count'))
    plt.tight_layout()
    _savefig(fig, os.path.join(PLOT_DIR, '04_confidence_dist.png'))

def plot_summary_dashboard(history, y_te, y_pred, y_prob, tflite_kb, n_params, device_str):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(_t('智能正姿衣 · 体态识别模型训练报告 [TF版]', 'Smart Posture Shirt · TF Training Report'),
                 fontsize=15, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ep = range(1, len(history.history['accuracy']) + 1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep, history.history['accuracy'],     color=COLORS[0], label=_t('训练', 'Train'))
    ax1.plot(ep, history.history['val_accuracy'], color=COLORS[1], label=_t('验证', 'Val'))
    ax1.set_title(_t('准确率曲线', 'Accuracy')); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(alpha=0.3); ax1.spines[['top', 'right']].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ep, history.history['loss'],     color=COLORS[2], label=_t('训练', 'Train'))
    ax2.plot(ep, history.history['val_loss'], color=COLORS[3], label=_t('验证', 'Val'))
    ax2.set_title(_t('损失曲线', 'Loss')); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(alpha=0.3); ax2.spines[['top', 'right']].set_visible(False)

    ax3 = fig.add_subplot(gs[0, 2])
    cm_pct = confusion_matrix(y_te, y_pred).astype(float)
    cm_pct = cm_pct / cm_pct.sum(axis=1, keepdims=True) * 100
    ax3.imshow(cm_pct, cmap='Blues')
    ax3.set_xticks(range(N_CLASSES)); ax3.set_yticks(range(N_CLASSES))
    ax3.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax3.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax3.set_title(_t('混淆矩阵 (%)', 'Confusion (%)'))
    ax3.set_xlabel(_t('预测', 'Pred')); ax3.set_ylabel(_t('真实', 'True'))
    thresh = cm_pct.max() / 2
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax3.text(j, i, f'{cm_pct[i, j]:.0f}%', ha='center', va='center',
                     fontsize=9, color='white' if cm_pct[i, j] > thresh else 'black')

    ax4 = fig.add_subplot(gs[1, 0])
    f1s  = f1_score(y_te, y_pred, average=None)
    bars = ax4.bar(CLASS_NAMES, f1s, color=COLORS, edgecolor='white')
    ax4.set_ylim(0, 1.15); ax4.set_title(_t('各类别 F1 Score', 'F1 Score per Class'))
    for bar, v in zip(bars, f1s):
        ax4.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                 f'{v:.3f}', ha='center', fontsize=9)
    ax4.axhline(0.9, color='gray', linestyle='--', alpha=0.5, label='0.9 baseline')
    ax4.legend(fontsize=8); ax4.spines[['top', 'right']].set_visible(False)

    ax5 = fig.add_subplot(gs[1, 1])
    ok   = y_pred == y_te
    conf = y_prob.max(axis=1)
    ax5.hist(conf[ok],  bins=20, alpha=0.75, color=COLORS[2], label=f'{_t("正确", "Correct")} n={ok.sum()}')
    ax5.hist(conf[~ok], bins=20, alpha=0.75, color=COLORS[3], label=f'{_t("错误", "Wrong")} n={(~ok).sum()}')
    ax5.set_title(_t('最大置信度分布', 'Max Confidence Dist'))
    ax5.set_xlabel(_t('置信度', 'Confidence')); ax5.set_ylabel(_t('样本数', 'Count'))
    ax5.legend(); ax5.spines[['top', 'right']].set_visible(False)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off'); ax6.set_title(_t('模型指标', 'Model Metrics'), fontweight='bold', pad=10)
    size_status = '✅' if tflite_kb < 100 else '❌ OVER'
    metrics = [
        (_t('测试准确率',     'Test Accuracy'),  f'{(y_pred == y_te).mean():.2%}'),
        (_t('最佳验证准确率', 'Best Val Acc'),   f'{max(history.history["val_accuracy"]):.2%}'),
        (_t('训练 Epochs',    'Epochs'),         str(len(ep))),
        (_t('模型参数量',     'Params'),         f'{n_params:,}'),
        ('TFLite (INT8)',                         f'{tflite_kb:.1f} KB  {size_status}'),
        (_t('训练设备',       'Device'),         device_str),
        (_t('TF 版本',       'TF Version'),      tf.__version__),
        (_t('采样率',         'Sample Rate'),    f'{SAMPLE_RATE} Hz'),
        (_t('目标芯片',       'Target MCU'),     'ESP32-C6'),
    ]
    for i, (k, v) in enumerate(metrics):
        y_pos = 0.92 - i * 0.095
        ax6.text(0.05, y_pos, k, transform=ax6.transAxes, fontsize=9, color='gray')
        ax6.text(0.95, y_pos, v, transform=ax6.transAxes, fontsize=9.5,
                 fontweight='bold', ha='right', color=COLORS[0] if i == 0 else 'black')
        ax6.axhline(y_pos - 0.03, xmin=0.05, xmax=0.95, color='#eeeeee', linewidth=0.8)

    _savefig(fig, os.path.join(PLOT_DIR, '05_summary_dashboard.png'))

# ══════════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════════
def main(csv_folder: str = CSV_FOLDER):
    """
    主训练流程
    :param csv_folder: 数据集文件夹路径
    """
    # 创建输出目录
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 设备标识
    device_str = "GPU (CUDA)" if _tf_gpu else "CPU"
    if _FORCE_CPU:
        device_str = f"CPU (GPU {_gpu_info['name']} 不兼容 TF，已强制回退)"

    print("=" * 60)
    print("智能正姿衣 · 1D-CNN 体态识别模型训练（TensorFlow 纯净版）")
    print(f"  采样率: {SAMPLE_RATE}Hz | 窗口: {WIN}点({WINDOW_SEC}s) | 步长: {STRIDE}点")
    print(f"  训练设备: {device_str}")
    print(f"  TF 版本: {tf.__version__}")
    print(f"  数据集路径: {os.path.abspath(csv_folder)}")
    print(f"  输出路径: {os.path.abspath(OUT_DIR)}")
    print("=" * 60)

    t0 = time.time()

    try:
        # ── [1/5] 加载数据 ──
        print("\n[1/5] 加载数据...")
        X_raw, y_raw = load_data(csv_folder)

        # ── [2/5] 数据分割与滑动窗口 ──
        print("\n[2/5] 数据分割与滑动窗口...")
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X_raw, y_raw, test_size=0.3, stratify=y_raw, random_state=42)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

        X_tr,  y_tr  = sliding_window(X_tr,  y_tr)
        X_val, y_val = sliding_window(X_val, y_val)
        X_te,  y_te  = sliding_window(X_te,  y_te)

        print(f"  训练集: {X_tr.shape}  验证集: {X_val.shape}  测试集: {X_te.shape}")

        # 可视化类别分布
        plot_class_distribution(y_tr, y_val, y_te)

        # 归一化（仅用训练集统计，避免数据泄露）
        mean = X_tr.mean(axis=(0, 1), keepdims=True)
        std  = X_tr.std(axis=(0, 1),  keepdims=True) + 1e-8
        X_tr  = (X_tr  - mean) / std
        X_val = (X_val - mean) / std
        X_te  = (X_te  - mean) / std

        # 保存归一化参数
        mean_flat = mean.flatten().astype(np.float32)
        std_flat  = std.flatten().astype(np.float32)
        np.save(os.path.join(OUT_DIR, 'norm_mean.npy'), mean)
        np.save(os.path.join(OUT_DIR, 'norm_std.npy'),  std)
        np.save(os.path.join(OUT_DIR, 'norm_mean_flat.npy'), mean_flat)
        np.save(os.path.join(OUT_DIR, 'norm_std_flat.npy'),  std_flat)
        print(f"  归一化参数已保存")
        print(f"  Mean: {np.array2string(mean_flat, precision=4, separator=', ')}")
        print(f"  Std:  {np.array2string(std_flat, precision=4, separator=', ')}")

        # ── [3/5] 构建并训练模型 ──
        print("\n[3/5] 构建并训练模型...")
        model = build_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()

        # 训练
        t_train_start = time.time()
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=60,
            batch_size=128,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=12, restore_best_weights=True, verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    patience=6, factor=0.5, min_lr=1e-6, verbose=1
                ),
            ]
        )
        t_train = time.time() - t_train_start
        print(f"  ⏱  训练耗时: {t_train:.1f}s ({t_train/60:.1f}min)")

        # 保存 Keras 模型
        keras_path = os.path.join(OUT_DIR, 'posture_model.keras')
        model.save(keras_path)
        print(f"  ✅ Keras 模型已保存至: {keras_path}")

        # ── [4/5] 模型评估与可视化 ──
        print("\n[4/5] 模型评估与可视化...")
        plot_training_history(history)

        test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
        y_prob = model.predict(X_te, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        print(f"\n  测试集损失: {test_loss:.4f} | 测试集准确率: {test_acc:.4f}")
        print("\n" + classification_report(y_te, y_pred, target_names=CLASS_NAMES, digits=4))

        plot_confusion_matrix(y_te, y_pred)
        plot_confidence_distribution(y_te, y_prob)

        # ── [5/5] 导出 TFLite (INT8 全量化) ──
        print("\n[5/5] 导出 TFLite (INT8全量化)...")
        n_rep = min(500, len(X_tr))
        rep_indices = np.random.choice(len(X_tr), n_rep, replace=False)

        def rep_data():
            for idx in rep_indices:
                yield [X_tr[idx:idx + 1].astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations              = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset     = rep_data
        converter.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type       = tf.int8
        converter.inference_output_type      = tf.int8

        tflite_model = converter.convert()
        tflite_kb   = len(tflite_model) / 1024

        tflite_path = os.path.join(OUT_DIR, 'posture_model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"  ✅ TFLite 模型已保存至: {tflite_path} (大小: {tflite_kb:.1f} KB)")

        # 生成 C 头文件
        c_header = tflite_to_c_header(tflite_model, mean_flat=mean_flat, std_flat=std_flat)
        c_header_path = os.path.join(OUT_DIR, 'posture_model.h')
        with open(c_header_path, 'w', encoding='utf-8') as f:
            f.write(c_header)
        print(f"  ✅ C 头文件已保存至: {c_header_path}")

        # 汇总仪表盘
        plot_summary_dashboard(history, y_te, y_pred, y_prob, tflite_kb, model.count_params(), device_str)

        # ── 最终报告 ──
        t_total = time.time() - t0
        print("\n" + "=" * 60)
        print("✅ 训练流程全部完成！")
        print(f"  训练设备:       {device_str}")
        print(f"  最终测试准确率:  {test_acc:.2%}")
        print(f"  TFLite 模型大小: {tflite_kb:.1f} KB  ({'✅ 达标 (<100KB)' if tflite_kb < 100 else '❌ 超限'})")
        print(f"  模型总参数量:    {model.count_params():,}")
        print(f"  训练耗时:       {t_train:.1f}s ({t_train/60:.1f}min)")
        print(f"  总耗时:         {t_total:.1f}s ({t_total/60:.1f}min)")
        print(f"  所有输出文件位于: {os.path.abspath(OUT_DIR)}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 训练过程出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()