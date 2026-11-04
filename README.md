# ESP32-TinyML

> 面向 ESP32 的端侧 AI（TinyML）学习笔记。
> 仓库按**分支**组织：`main` 只是索引页，完整内容在各自的分支上。

---

## 两条路线

| 平台 | 技术路线 | 分支 | 内容概览 |
|---|---|---|---|
| **ESP32-P4** | esp-dl + esp-ppq（Espressif 自研运行时） | [esp32-p4](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-p4) | esp-dl 量化与部署、自定义算子、esp-who 框架、P4 平台规格与 RISC-V/PIE |
| **ESP32-C5** | esp-nn + TFLite Micro（通用生态路线） | [esp32-c5](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-c5) | esp-nn/TFLM 运行时、手势识别实战、软浮点、性能与量化对比 |

## 公共基础

两份笔记共享的概念部分统一维护在主分支：

- **[模型与部署基础（basics.md）](basics.md)** —— 模型 = 结构 + 参数、推理 vs 训练、张量与内存布局、常见模型文件格式、量化概念、FlatBuffers、初学常识校正。
- **[神经网络基础（neural-networks.md）](neural-networks.md)** —— 神经元、激活函数、层与算子、全连接网络（MLP）、卷积神经网络（卷积核/通道/感受野）、RNN/LSTM/TCN，以及术语速查表。

> 推荐顺序：**neural-networks.md**（网络结构）→ **basics.md**（量化与部署）→ 对应平台分支（实战）。

两个分支的「第 2 章」都只保留平台特有内容，公共概念统一指回这里，避免同一份知识维护两遍。

## 快速开始

```bash
git clone https://github.com/yh-xh/ESP32-TinyML.git
cd ESP32-TinyML

git checkout esp32-p4   # ESP32-P4 / esp-dl 笔记
git checkout esp32-c5   # ESP32-C5 / esp-nn 笔记
```

## 两条路线怎么选

完整的 8 维对比（模型格式、量化工具、运行时、算子来源、模型来源、芯片支持、优劣势）见
**[basics.md · 两条实现路线](basics.md#8-两条实现路线)**。一句话结论：

- 想直接跑现成的 TFLite 模型、看重跨平台、上手快 → **esp-nn + TFLM（C5）**
- 做人脸/检测、想用 esp-ppq 深度调量化、吃 P4 的 SIMD → **esp-dl + esp-ppq（P4）**

> 两条路线与 TinyML 的层级关系，在各分支第 1 章也有平台视角的说明。

## 说明

- 两个分支的 git 历史相互独立（各自从原仓库迁移而来），不需要 merge，也不会互相影响；
- 笔记持续更新，内容以各分支的 README 为准；
- 原 `esp32-p4-notes` / `esp32-c5-notes` 仓库已合并到这里。
