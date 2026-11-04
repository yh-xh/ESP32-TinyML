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

两份笔记共享的概念部分已抽离成一份，统一维护在主分支：

**[模型与部署基础（basics.md）](basics.md)** —— 模型 = 结构 + 参数、推理 vs 训练、张量与内存布局、常见模型文件格式、量化概念、FlatBuffers、初学常识校正。

两个分支的「第 2 章」都只保留平台特有内容，公共概念统一指回这里，避免同一份知识维护两遍。

## 快速开始

```bash
git clone https://github.com/yh-xh/ESP32-TinyML.git
cd ESP32-TinyML

git checkout esp32-p4   # ESP32-P4 / esp-dl 笔记
git checkout esp32-c5   # ESP32-C5 / esp-nn 笔记
```

## 两条路线怎么选

| 维度 | esp-dl + esp-ppq（P4） | esp-nn + TFLM（C5） |
|---|---|---|
| 模型格式 | `.espdl` | `.tflite` |
| 量化工具 | esp-ppq（KL/percentile、TQT、AutoQuant） | TFLite Converter |
| 运行时 | esp-dl execution plan | TFLM 解释器 + esp-nn kernel |
| 算子来源 | esp-dl 自己的 Module + P4 SIMD | esp_nn kernel + TFLM 参考实现 |
| 适合场景 | 检测/识别、想深度调量化、吃 P4 的 SIMD | 通用 TFLite 生态、看重跨平台 |
| 笔记入口 | [esp32-p4 分支](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-p4) | [esp32-c5 分支](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-c5) |

> 两条路线的关系（TinyML / TFLM / esp_nn / esp-dl 的层级）在 P4 笔记第 1 章和 C5 笔记第 1 章都有详细说明。

## 说明

- 两个分支的 git 历史相互独立（各自从原仓库迁移而来），不需要 merge，也不会互相影响；
- 笔记持续更新，内容以各分支的 README 为准；
- 原 `esp32-p4-notes` / `esp32-c5-notes` 仓库已合并到这里。
