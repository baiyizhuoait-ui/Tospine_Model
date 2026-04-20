# Tospine_Model · 智能正姿衣 体态识别模型

> 面向 ESP32-C6 的轻量级 1D-CNN 体态识别系统，支持 4 类姿态实时分类，模型以 INT8 全量化 TFLite 格式部署。

---

## 项目简介

**智能正姿衣（Smart Posture Shirt）** 是一套嵌入式体态监测方案，通过衣物上的 3 个 FSR 柔性压力传感器与 BMI160 六轴 IMU，以 50 Hz 采样率持续采集穿戴者的姿态数据，并在 ESP32-C6 微控制器上实时推理，识别以下 4 种体态：

| 标签 | 名称 | 说明 |
|------|------|------|
| 0 | 挺直 | 标准坐姿 |
| 1 | 驼背 | 腰背前屈 |
| 2 | 左低右高 | 左肩偏低 |
| 3 | 左高右低 | 右肩偏低 |

**当前最佳结果：** 测试准确率 **90%**，TFLite 模型大小 **22.4 KB**（目标 ≤ 40 KB）。

---

## 硬件平台

| 组件 | 规格 |
|------|------|
| 微控制器 | ESP32-C6 (RISC-V, 160 MHz) |
| 内存 | 512 KB SRAM / 4 MB Flash |
| 传感器 | FSR 柔性压力 ×3 + BMI160 六轴 IMU |
| 采样率 | 50 Hz（定时器同步） |
| 推理运行时 | TFLite Micro |

---

## 仓库结构

```
Tospine_Model/
│
├── WSL版_train_posture_pytorch.py   # 主训练脚本（WSL 本地，PyTorch GPU + TFLite 导出）
│
├── Colab版_TF_Keras_CNN_4_19.ipynb  # Colab 笔记本 —— TensorFlow/Keras 原始版
├── Colab版_PyTorch_4_20.ipynb       # Colab 笔记本 —— PyTorch 迁移版（含 tqdm 进度条）
│
├── output/
│   └── posture_model.h              # 已生成的 C 头文件（可直接用于 ESP32 工程）
│
└── .gitignore
```

### 版本对比

| 文件 | 框架 | 运行环境 | 适用场景 |
|------|------|----------|----------|
| `WSL版_train_posture_pytorch.py` | PyTorch → TFLite | WSL / Linux 本地 | 有 GPU 的本地训练 |
| `Colab版_TF_Keras_CNN_4_19.ipynb` | TensorFlow / Keras | Google Colab | 免配置云端训练（原始版） |
| `Colab版_PyTorch_4_20.ipynb` | PyTorch → TFLite | Google Colab | 云端训练 + 英文 Dashboard |

---

## 模型架构

```
输入 (100, 9)          ← 2s 窗口 × 9 通道
  │
Conv1D(16, k=5) + ReLU + MaxPool → (50, 16)
Conv1D(32, k=3) + ReLU + MaxPool → (25, 32)
Conv1D(64, k=3) + ReLU           → (25, 64)
GlobalAveragePooling1D            → (64,)
Dense(32) + ReLU + Dropout(0.3)
Dense(4)  + Softmax               → 4 类概率
```

**参数量：** ~10,700 | **TFLite INT8 大小：** 22.4 KB

---

## 快速开始

### WSL 本地训练

```bash
# 激活 gpu_env（内含 PyTorch CUDA + TensorFlow）
source ~/workspace/tf_env/bin/activate

# 确认 GPU 可用
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 修改脚本顶部的数据路径，然后运行
python WSL版_train_posture_pytorch.py
```

### Google Colab 训练

1. 打开 `Colab版_TF_Keras_CNN_4_19.ipynb` 或 `Colab版_PyTorch_4_20.ipynb`
2. 将训练数据 zip 上传到 Google Drive（路径：`MyDrive/posture_synthetic_data.zip`）
3. 依次运行所有单元格

### 数据格式要求

每个 CSV 文件需包含以下列（顺序不限）：

```
time, fsr_left, fsr_right, fsr_spine, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, label
```

采样率 50 Hz，`label` 取值 0–3。

---

## 训练超参数（v2）

| 参数 | 值 | 说明 |
|------|----|------|
| Conv 通道 | 16 / 32 / 64 | v2 从 8/16/32 扩容 |
| Dropout | 0.3 | v2 从 0.4 调低 |
| 初始学习率 | 5e-4 | v2 从 1e-3 调低，抑制 val loss 抖动 |
| Batch Size | 128 | — |
| 最大 Epochs | 60 | EarlyStopping patience=12 |
| 类别权重 | 1/count | 补偿 Upright 样本偏多 |
| 窗口 | 100 点（2s） | 滑动步长 50 点（50% overlap） |

---

## 部署到 ESP32-C6

训练完成后，`output/` 目录下会生成：

```
posture_model.tflite   ← 烧录到 Flash
posture_model.h        ← 复制到 ESP32 Arduino/ESP-IDF 工程 ★
posture_model.pt       ← PyTorch 权重（本地微调备用）
```

在 ESP32 固件中，推理前需按 `posture_model.h` 内的 `norm_mean` / `norm_std` 对原始传感器数据做 Z-score 归一化：

```c
// 预处理示例（9 通道）
for (int i = 0; i < 9; i++) {
    input_data[i] = (raw_data[i] - norm_mean[i]) / norm_std[i];
}
```

---

## 版本记录

| Tag | 准确率 | TFLite | 说明 |
|-----|--------|--------|------|
| v2.0 | 89.61% | 22.4 KB | Conv 16/32/64，类别权重，LR 5e-4 |
| v1.0 | ~89% | 13.9 KB | Conv 8/16/32，基线版本 |

---

## 项目背景

本项目为学生竞赛作品，基于马来西亚本地真实采集数据开发，目标是以极低硬件成本实现穿戴式姿态监测，辅助用户改善长期不良坐姿习惯。
