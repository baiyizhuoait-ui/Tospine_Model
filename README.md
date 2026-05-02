# Tospine_Model · 智能正姿衣 体态识别模型

> 面向 ESP32-C6 的轻量级 1D-CNN 体态识别系统，支持 4 类姿态实时分类，模型以 INT8 全量化 TFLite 格式部署。

---

## 项目简介

**智能正姿衣（Smart Posture Shirt）** 是一套嵌入式体态监测方案，通过衣物上的 3 个 Flex 弯曲传感器（左肩 Flex 2.2、右肩 Flex 2.2、背部 Flex 4.5）与 BMI160 六轴 IMU，以 50 Hz 采样率持续采集穿戴者体态数据，实时推理出坐姿状态（4 分类），通过振动反馈纠正用户不良坐姿。

### 姿态分类

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
| 弯曲传感器 | Flex 2.2 ×2（左肩、右肩）+ Flex 4.5 ×1（背部） |
| 电压输出范围 | 1.300 V – 1.600 V（分辨率 1 mV） |
| IMU | BMI160 六轴（3轴加速度 + 3轴陀螺仪） |
| 采样率 | 50 Hz（定时器同步） |
| 推理运行时 | TFLite Micro |

> **传感器说明：** Flex 传感器输出模拟电压信号，弯曲角度变化对应约 10–20 mV 的电压变化。传感器重复性误差约 ±5–15 mV，训练时建议对输入添加高斯噪声增强以提升鲁棒性。

---

## 仓库结构

```
Tospine_Model/
│
├── README.md                          # 项目文档
├── .gitignore                         # Git 忽略配置
│
├── 📁 code/                           # 训练脚本集合
│   ├── WSL pytorch版.py               # WSL 本地训练 - PyTorch + TFLite 导出
│   ├── WSL 纯tensorflow版.py          # WSL 本地训练 - 纯 TensorFlow 实现
│   ├── pytorch版.py                   # 基础 PyTorch 训练脚本
│   ├── 纯tensorflow版.py              # 基础 TensorFlow 训练脚本
│   ├── COLAB pytorch版.ipynb          # Colab 云端训练 - PyTorch 版（推荐）
│   └── COLAB 纯tensorflow版.ipynb     # Colab 云端训练 - TensorFlow 版
│
├── 📁 output/                         # 模型输出与可视化结果
│   ├── 🧠 posture_model.tflite        # INT8 全量化 TFLite 模型（22.4 KB）⭐
│   ├── 🧠 posture_model.h             # C 头文件（ESP32 直接使用）⭐
│   ├── 🧠 posture_model.pt            # PyTorch 权重文件（本地微调用）
│   ├── 📊 norm_mean_flat.npy          # 归一化均值（9 通道）
│   ├── 📊 norm_std_flat.npy           # 归一化标准差（9 通道）
│   │
│   └── 📁 plots/                      # 训练可视化图表
│       ├── 01_class_distribution.png  # 类别样本分布
│       ├── 02_training_history.png    # Loss / Accuracy 曲线
│       ├── 03_confusion_matrix.png    # 测试集混淆矩阵
│       ├── 04_confidence_dist.png     # 预测置信度分布
│       └── 05_summary_dashboard.png   # 综合性能仪表板
│
└── 📁 posture_data/                   # 训练数据集
    └── posture_synthetic_data/        # 合成训练数据（4000 CSV 文件）
        ├── upright/                   # 类别 0：挺直（~1000 样本）
        ├── slouch/                    # 类别 1：驼背（~1000 样本）
        ├── left_low_right_high/       # 类别 2：左低右高（~1000 样本）
        └── left_high_right_low/       # 类别 3：左高右低（~1000 样本）
```

### 文件说明详表

#### 📂 code/ - 训练脚本

| 文件 | 框架 | 环境 | 特点 | 何时使用 |
|------|------|------|------|---------|
| `WSL pytorch版.py` | PyTorch | WSL/Linux + GPU | 完整功能，带进度条和可视化 | ✅ 推荐本地训练 |
| `WSL 纯tensorflow版.py` | TensorFlow | WSL/Linux + GPU | 原生 TF 实现 | 已有 TF 环境 |
| `pytorch版.py` | PyTorch | Python | 简化版，仅核心逻辑 | 测试/学习用 |
| `纯tensorflow版.py` | TensorFlow | Python | 简化 TF 版本 | 测试/学习用 |
| `COLAB pytorch版.ipynb` | PyTorch | Google Colab | 云端免费 GPU，英文界面 | ✅ 推荐云端训练 |
| `COLAB 纯tensorflow版.ipynb` | TensorFlow | Google Colab | TF Keras 原始版本 | 云端 TF 首选 |

#### 📂 output/ - 部署产物

| 文件 | 大小 | 用途 | 说明 |
|------|------|------|------|
| `posture_model.tflite` | 22.4 KB | **🎯 ESP32 部署** | INT8 全量化，直接烧录到 Flash |
| `posture_model.h` | ~142 KB | **🎯 ESP32 集成** | C 头文件，含量化权重 + 归一化参数 |
| `posture_model.pt` | ~46 KB | 本地微调 | PyTorch 权重，支持继续训练 |
| `norm_mean_flat.npy` | 164 B | 预处理 | 9 通道 Z-score 均值 |
| `norm_std_flat.npy` | 164 B | 预处理 | 9 通道 Z-score 标准差 |
| `plots/*` | 各异 | 分析评估 | 5 张训练过程与性能可视化图 |

#### 📂 posture_data/ - 训练数据

- **来源**：综合传感器真实数据 + 数据增强生成
- **数量**：4000 CSV 文件（4 类 × 1000 样本）
- **采样率**：50 Hz
- **时长**：100 点 = 2 秒窗口
- **特征**：9 通道（3×Flex弯曲传感器 + 6×IMU）

---

## 快速开始

### 📌 本地训练（WSL / Linux）

```bash
# 1. 激活虚拟环境
source ~/workspace/tf_env/bin/activate

# 2. 验证 GPU 可用性
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 3. 运行训练脚本
python code/WSL\ pytorch版.py
```

### ☁️ Colab 云端训练（推荐无 GPU 用户）

1. 打开 `code/COLAB pytorch版.ipynb`
2. 将训练数据 zip 上传至 Google Drive：`MyDrive/posture_synthetic_data.zip`
3. 依次执行所有单元格

### 📥 数据格式要求

每个 CSV 文件需包含以下列（顺序不限）：

```
time, flex_left, flex_right, flex_back, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, label
```

| 列名 | 传感器来源 | 说明 |
|------|-----------|------|
| `flex_left` | 左肩 Flex 2.2 | 电压值（V），范围约 1.300–1.600 |
| `flex_right` | 右肩 Flex 2.2 | 电压值（V），范围约 1.300–1.600 |
| `flex_back` | 背部 Flex 4.5 | 电压值（V），范围约 1.300–1.600 |
| `acc_x/y/z` | BMI160 加速度计 | 单位 g |
| `gyr_x/y/z` | BMI160 陀螺仪 | 单位 °/s |
| `label` | — | 0–3（4 分类） |

- **采样率**：50 Hz
- **标签**：0–3（4 分类）
- **窗口**：100 点（2 秒）

> ⚠️ **注意**：训练脚本中的 `SENSOR_COLS` 需与实际 CSV 列名一致。若硬件组输出的列名不同，修改脚本顶部的 `SENSOR_COLS` 列表即可，模型结构无需任何改动。

---

## 模型架构

```
输入 (100, 9)          ← 2s 窗口 × 9 通道（3×Flex + 6×IMU）
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

### 步骤 1：获取模型文件

训练完成后，`output/` 目录下会生成：

```
posture_model.tflite   ← 烧录到 Flash
posture_model.h        ← 复制到 ESP32 Arduino/ESP-IDF 工程 ★
posture_model.pt       ← PyTorch 权重（本地微调备用）
```

### 步骤 2：集成到 ESP32 固件

将 `posture_model.h` 复制到 ESP32 工程的 `include/` 目录。

### 步骤 3：数据预处理

推理前需完成以下两步预处理：

**① Flex 传感器通道归一化**（建议用训练集 min/max 线性映射）：
```c
// Flex 通道线性归一化到 [0, 1]
// flex_min / flex_max 从训练数据统计得到，烧录为常量
float flex_norm = (raw_voltage - flex_min[ch]) / (flex_max[ch] - flex_min[ch]);
```

**② Z-score 归一化**（全通道，使用 posture_model.h 内参数）：
```c
// 预处理示例（9 通道）
for (int i = 0; i < 9; i++) {
    input_data[i] = (raw_data[i] - norm_mean[i]) / norm_std[i];
}

// 运行推理
model_predict(input_data, output_probs);

// 获取预测类别
int predicted_class = argmax(output_probs);
```

---

## 版本记录

| Tag | 准确率 | TFLite 大小 | 说明 |
|-----|--------|-----------|------|
| v2.0 | 89.61% → **90%** | 22.4 KB | Conv 16/32/64，类别权重，LR 5e-4；传感器改为 Flex×3 + BMI160 |
| v1.0 | ~89% | 13.9 KB | Conv 8/16/32，基线版本；FSR×3 + BMI160 |

---

## 项目背景

本项目为学生竞赛作品，基于马来西亚本地真实采集数据开发，目标是以极低硬件成本实现穿戴式姿态监测，辅助用户改善长期不良坐姿习惯。
