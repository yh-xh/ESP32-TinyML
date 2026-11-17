# ESP32-TinyML

> 面向 ESP32 的端侧 AI（TinyML）学习笔记。
> 仓库按**分支**组织：`main` 只是索引页，完整内容在各自的分支上。

---

## 两条路线

| 平台 | 技术路线 | 分支 | 内容概览 |
|---|---|---|---|
| **ESP32-P4** | esp-dl + esp-ppq（Espressif 自研运行时） | [esp-dl-esp-ppq](https://github.com/yh-xh/ESP32-TinyML/tree/esp-dl-esp-ppq) | esp-dl 量化与部署、自定义算子、esp-who 框架、P4 平台规格与 RISC-V/PIE |
| **ESP32-C5 / ESP32-S3（XIAO Sense）** | TFLite Micro + esp-nn（通用生态路线） | [tflite-micro-esp-nn](https://github.com/yh-xh/ESP32-TinyML/tree/tflite-micro-esp-nn) | TFLM/esp-nn 运行时、手势识别、人形检测、从重训练到部署、实测验证与踩坑 |

## 公共基础

两条路线共享的概念部分统一维护在主分支：

- **[模型与部署基础（model-deployment.md）](model-deployment.md)** —— 模型 = 结构 + 参数、模型的作用、架构/权重/框架的区别、推理 vs 训练、张量与内存布局、常见模型文件格式、量化概念、训练/校准/量化的区别、四种数据集的分工、云端/边缘/端侧的算力分工、TinyML 与 AIoT、模型压缩（蒸馏/剪枝/量化）、FlatBuffers、初学常识校正。
- **[神经网络基础（neural-networks.md）](neural-networks.md)** —— 从感知机到深度学习的简史讲起，厘清 AI / ML / DL 的关系，再用一条流水线讲清网络结构（特征提取 → 融合压缩 → 决策），逐个认识神经元、激活函数、卷积核、池化、BN、残差等零件，最后把概念放回分类/检测/时序三类真实网络，讲清训练循环，并用一个 2→2→1 的小网络手算一次完整训练（前向 / 损失 / 反向 / 更新）。以直觉和类比为主，只在手算时用一点必要数学。

> 推荐顺序：**neural-networks.md**（网络结构）→ **model-deployment.md**（量化与部署）→ 对应平台分支（实战）。

两条路线的「第 2 章」都只保留平台特有内容，公共概念统一指回这里，避免同一份知识维护两遍。

## 快速开始

```bash
git clone https://github.com/yh-xh/ESP32-TinyML.git
cd ESP32-TinyML

git checkout esp-dl-esp-ppq       # ESP32-P4 / esp-dl + esp-ppq 笔记
git checkout tflite-micro-esp-nn  # ESP32-C5 / ESP32-S3 / TFLite Micro + esp-nn 笔记
```

## 两条路线怎么选

完整的 8 维对比（模型格式、量化工具、运行时、算子来源、模型来源、芯片支持、优劣势）见
**[模型与部署基础 · 两条实现路线](model-deployment.md#8-两条实现路线)**。一句话结论：

- 想直接跑现成的 TFLite 模型、看重跨平台、上手快 → **TFLite Micro + esp-nn（C5/S3）**
- 做人脸/检测、想用 esp-ppq 深度调量化、吃 P4 的 SIMD → **esp-dl + esp-ppq（P4）**

> 两条路线与 TinyML 的层级关系，在各分支第 1 章也有平台视角的说明。

## 说明

- `esp-dl-esp-ppq` 与 `tflite-micro-esp-nn` 的 git 历史相互独立；后者由原 `esp32-c5` 和 `xiao_esp32-s3` 笔记合并而来；
- 笔记持续更新，内容以各分支的 README 为准；
- 原 `esp32-p4-notes` / `esp32-c5-notes` 仓库已合并到这里；原 `esp32-p4` / `esp32-c5` / `xiao_esp32-s3` 分支已改名为上述两条路线分支。
