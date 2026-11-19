# TensorFlow 与全家桶：从训练到 ESP32 部署

> **适用环境**：TensorFlow 2.15.1 / Keras 2.15 / TFLite Micro 1.4.3 / esp-nn 1.5.0 / ESP-IDF 5.5.3
> **记录时间**：2026-11
> 本文不教 Keras API，而是把 **TensorFlow、Keras、TFLite、TFLite Micro、esp-tflite-micro、esp-nn** 放在一张图里，讲清它们各自是什么、边界在哪、什么时候该用哪个。
> 配套实战见 [README.md](README.md)；模型与部署基础见主分支 [model-deployment.md](https://github.com/yh-xh/ESP32-TinyML/blob/main/model-deployment.md)。

---

## 目录

- [1. 先看全景](#1-先看全景)
- [2. TensorFlow 核心是什么](#2-tensorflow-核心是什么)
- [3. Keras 是什么](#3-keras-是什么)
- [4. tf.data：训练数据管道](#4-tfdata训练数据管道)
- [5. TensorBoard：训练可视化](#5-tensorboard训练可视化)
- [6. Model Optimization：剪枝/量化/QAT](#6-model-optimization剪枝量化qat)
- [7. 预训练资源：Hub / Datasets / Model Garden](#7-预训练资源hub--datasets--model-garden)
- [8. TFLite / LiteRT：部署格式与完整运行时](#8-tflite--litert部署格式与完整运行时)
- [9. TFLite Micro：MCU 运行时](#9-tflite-microcu-运行时)
- [10. esp-tflite-micro 与 esp-nn](#10-esp-tflite-micro-与-esp-nn)
- [11. 模型格式与产物对比](#11-模型格式与产物对比)
- [12. 周边组件：什么时候用，什么时候不用](#12-周边组件什么时候用什么时候不用)
- [13. 你的最小栈](#13-你的最小栈)
- [14. 常见误区](#14-常见误区)
- [15. 学习路径](#15-学习路径)
- [16. 参考](#16-参考)

---

## 1. 先看全景

```text
TensorFlow 生态
│
├── 训练侧（PC）
│   ├── Keras（tf.keras / Keras 3）
│   ├── tf.data
│   ├── TensorBoard
│   ├── Model Optimization（剪枝 / QAT / 聚类）
│   └── TF Datasets / Hub / Model Garden
│
├── 部署侧
│   ├── 模型文件：.keras / .h5 / SavedModel
│   ├── TFLite（LiteRT）
│   │   ├── TFLite Converter   ← 生成 .tflite
│   │   ├── TFLite Interpreter ← PC/手机/边缘设备推理
│   │   └── Delegates（XNNPACK / GPU / NNAPI…）
│   └── TFLite Micro（TFLM）
│       ├── MicroInterpreter
│       ├── MicroMutableOpResolver
│       └── tensor_arena
│           └── esp-tflite-micro + esp-nn（ESP32）
│
└── 周边（多数与 MCU 无关）
    ├── TF Serving / TFX
    ├── TF.js
    ├── TF Probability / Agents / Federated
    ├── TF Ranking / Recommenders
    └── TF Graphics / Text / I/O
```

你实际要走的路线只有一条：

```text
Keras 训练
  → TFLite Converter
  → .tflite
  → esp-tflite-micro（TFLM）
  → esp-nn 加速
  → ESP32 推理
```

其余组件先知道“是什么、什么时候用”即可，不要为了全家桶而全家桶。

---

## 2. TensorFlow 核心是什么

**TensorFlow 是底层计算引擎和生态**，不是“训练框架”的同义词。

它提供：

| 概念 | 作用 |
|---|---|
| `tf.Tensor` | 多维数组 |
| `tf.Variable` | 可训练变量 |
| `tf.Operation` | 计算节点 |
| Eager execution | 立即执行，方便调试 |
| `tf.function` | 把 Python 函数追踪成图，提升性能/可导出 |
| `tf.Module` | 变量和函数的容器 |
| SavedModel | 包含图、权重、签名的部署格式 |
| 设备/分布式 | CPU/GPU/TPU、多机多卡 |

### 关键区别

- **Eager**：像 NumPy 一样逐行执行，调试方便；
- **Graph / `tf.function`**：先建图再执行，性能好，可导出；
- **TFLite Converter** 也需要先把 Keras 模型追踪成图，才能转成 `.tflite`。

大多数嵌入式开发者不需要直接写底层 `tf.Operation`，用 Keras 就够了。理解 TensorFlow 核心的价值在于：

- 知道“模型”最终是一张图 + 权重；
- 知道训练图和推理图的区别；
- 知道 SavedModel/`.tflite` 是怎么来的。

---

## 3. Keras 是什么

**Keras 是高层模型 API**，负责：

- 搭网络：`Sequential` / `Functional` / 自定义层；
- 训练：`compile` / `fit` / `evaluate`；
- 保存：`.keras` / `.h5` / SavedModel；
- 回调：EarlyStopping、ModelCheckpoint、TensorBoard。

在 TensorFlow 2.x 里：

```python
tf.keras
```

就是 TensorFlow 内置的 Keras API。

### Keras 2 vs Keras 3

| | Keras 2 | Keras 3 |
|---|---|---|
| 典型环境 | TensorFlow ≤ 2.15 | TensorFlow 2.16+ 默认 |
| 后端 | TensorFlow | TensorFlow / JAX / PyTorch |
| 保存格式 | `.h5` / SavedModel / `.keras` | `.keras` 为主 |
| TFLite 转换 | 最成熟 | 需要看版本，常用 `tf-keras` 兼容层 |
| 适合 | MCU 部署、稳定链路 | 通用深度学习、多后端 |

**你的选择**：

- 当前 `tf-esp` 环境是 TensorFlow 2.15.1 + Keras 2.15；
- 对“训练 → TFLite → ESP32”这条路，Keras 2 最稳；
- Keras 3 可以学，但不要作为第一次 MCU 部署的环境。

### Keras 与 TensorFlow 的关系

```text
TensorFlow = 引擎 + 生态
Keras      = 引擎上面的高层 API
tf.keras   = TensorFlow 里的 Keras
```

不要把 Keras 和 TensorFlow 当成两个互斥平台。选 Keras，通常还是要选一个后端；做 TFLite 部署时，后端选 TensorFlow。

---

## 4. tf.data：训练数据管道

`tf.data` 负责把原始数据变成高效的 batch 流：

```python
ds = tf.data.Dataset.from_tensor_slices((x, y))
ds = ds.shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)
```

它解决：

- 大数据不能一次性放进内存；
- 数据加载/预处理与 GPU/CPU 计算重叠；
- 复杂的 `map`、`cache`、`interleave`、`window`；
- 训练/验证/测试管道统一。

### 什么时候需要

- 图像/音频/文本数据集较大；
- 需要在线增强；
- 需要从 TFRecord/文件读取。

小规模 NumPy 数据直接用 `model.fit(x, y)` 就行，不必强行上 `tf.data`。

### 与 MCU 的关系

`tf.data` 只存在于 PC 训练侧；ESP32 上不跑 `tf.data`，端侧预处理通常是 C/C++ 手写。

---

## 5. TensorBoard：训练可视化

TensorBoard 是 TensorFlow 的训练可视化工具：

- loss/accuracy 曲线；
- 学习率、权重直方图；
- 计算图；
- embedding；
- profiling。

用法：

```python
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="logs")
model.fit(x, y, epochs=300, callbacks=[tensorboard_cb])
```

```bash
tensorboard --logdir logs
```

### 与 MCU 的关系

TensorBoard 只在 PC 训练阶段用；端侧性能要用 `esp_timer`、串口日志、`idf.py size`、`arena_used_bytes()` 等测量。

---

## 6. Model Optimization：剪枝/量化/QAT

TensorFlow Model Optimization Toolkit（TFMOT）提供训练期优化：

| 技术 | 作用 | 对 MCU 的价值 |
|---|---|---|
| 量化感知训练（QAT） | 训练时插入伪量化，模拟 int8 误差 | 精度不够时提升 int8 模型 |
| 剪枝 | 去掉不重要的权重 | 稀疏模型，但 MCU 运行时不一定受益 |
| 权重聚类 | 共享权重值，压缩模型 | 需要运行时支持 |
| 训练后量化（PTQ） | 不属于 TFMOT，属于 TFLite Converter | 最常用，简单快速 |

关键区别：

- **PTQ**：先训练 float32，再在 TFLite Converter 里量化；
- **QAT**：训练时就模拟量化，通常精度更好，但流程更复杂；
- 两者最终都生成 int8 `.tflite`，不是“训练 dtype 直接变 int8”。

对 ESP32 的建议：

1. 先做 PTQ；
2. 如果 int8 精度掉得厉害，再上 QAT；
3. 剪枝/聚类不是第一优先级，先看算子/内存/延迟。

---

## 7. 预训练资源：Hub / Datasets / Model Garden

| 资源 | 是什么 | 注意 |
|---|---|---|
| TensorFlow Datasets | 现成数据集 | 适合入门/基准 |
| TensorFlow Hub / Kaggle Models | 预训练模型 | 注意输入尺寸、算子、量化、许可证 |
| TensorFlow Model Garden | 官方模型实现 | 偏大，多数不适合 MCU |
| Keras 示例 | 模型代码 | 适合学习结构 |

**预训练模型不等于能上 MCU**。上 ESP32 前要检查：

- 模型多大，flash/PSRAM 是否放得下；
- 是否 int8 量化；
- 输入尺寸/预处理是否适合你的传感器；
- 用了哪些算子，TFLM 是否支持；
- 是否依赖 TFLite delegate / Flex ops；
- 许可证。

---

## 8. TFLite / LiteRT：部署格式与完整运行时

TFLite 现在官方文档归到 **LiteRT / Google AI Edge** 品牌下，但代码 API 仍是 `tf.lite`。

### 8.1 TFLite Converter

作用：把训练好的模型转成 `.tflite`。

输入可以是：

- Keras 模型对象；
- SavedModel；
- ConcreteFunction。

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

量化时：

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

### 8.2 TFLite Interpreter

作用：在 PC/手机/边缘设备上执行 `.tflite`。

```python
interp = tf.lite.Interpreter(model_path="model.tflite")
interp.allocate_tensors()
interp.set_tensor(input_index, data)
interp.invoke()
out = interp.get_tensor(output_index)
```

特点：

- 动态内存；
- 支持较多算子；
- 支持 delegate：XNNPACK、GPU、NNAPI、Core ML 等；
- 适合 PC 验证、手机/边缘设备部署。

### 8.3 TFLite 不是什么

- 不是训练框架；
- 不是 MCU 运行时；
- 不是“把 `.keras` 改后缀”就能得到的东西。

---

## 9. TFLite Micro：MCU 运行时

TFLite Micro（TFLM）是为微控制器设计的 **C++ 推理运行时**。它和完整 TFLite 是两套实现。

### 9.1 核心组件

| 组件 | 作用 |
|---|---|
| `tflite::GetModel` | 解析 `.tflite` FlatBuffer 指针 |
| `MicroMutableOpResolver` | 注册模型用到的算子 |
| `MicroInterpreter` | 调度算子、管理张量 |
| `tensor_arena` | 固定 RAM，所有中间张量复用 |
| `Invoke()` | 执行一次推理 |

### 9.2 TFLite vs TFLite Micro

| 维度 | TFLite（完整） | TFLite Micro |
|---|---|---|
| 目标 | PC / 手机 / 边缘设备 | MCU |
| 语言 | Python / C++ | C++ |
| 内存 | 动态分配 | 静态 arena |
| 算子 | 较多 | 子集，需注册 |
| Delegate | XNNPACK/GPU/NNAPI… | 不使用这些 delegate |
| 模型格式 | `.tflite` | 同一个 `.tflite` |
| 训练 | 不支持 | 不支持 |
| 体积 | 大 | 小 |

### 9.3 TFLM 的限制

- batch 只能是 1；
- 输入 shape 固定；
- 不支持动态 shape；
- 不支持 Flex / `SELECT_TF_OPS`；
- 算子要逐个注册；
- 内存预算要自己算；
- 量化模型要用 int8 才有 esp-nn 加速。

---

## 10. esp-tflite-micro 与 esp-nn

### 10.1 esp-tflite-micro

`espressif/esp-tflite-micro` 是 Espressif 对上游 TFLM 的移植/封装：

- 把上游 `tensorflow/lite/micro` 代码 vendored 进组件；
- 适配 ESP-IDF 构建；
- 提供 ESP32 系列示例；
- 依赖 `esp-nn`；
- 通过 IDF 组件管理器安装。

截至 2026-11，组件版本是 **1.4.3**，依赖 `esp-nn >= 1.4.2`，支持 ESP-IDF 5.1–6.0；本仓库使用 IDF 5.5.3。`esp-nn` 最新为 **1.5.0**。

你**不需要** clone 上游 `tensorflow/tflite-micro`，除非要读源码、改 runtime、加算子或给上游提 PR。

### 10.2 esp-nn

`esp-nn` 是 Espressif 的 **int8 算子库**：

- 不是运行时；
- 不解析模型；
- 被 esp-tflite-micro 调用；
- ESP32-S3 / P4 有汇编/SIMD 优化；
- ESP32 / C3 / C5 更多走通用优化；
- float32 模型不会走 esp-nn 的优化 kernel。

### 10.3 三者关系

```text
你的固件
  └── esp-tflite-micro（TFLM 运行时 + IDF 适配）
        └── esp-nn（int8 kernel）
              └── ESP32 硬件
```

| 名称 | 是什么 | 不是什么 |
|---|---|---|
| TFLM | MCU 推理运行时 | 不是 Python 包 |
| esp-tflite-micro | Espressif 的 TFLM 组件 | 不是另一套模型格式 |
| esp-nn | int8 算子库 | 不是运行时 |

---

## 11. 模型格式与产物对比

| 格式 | 谁产生 | 谁消费 | 内容 | 用途 |
|---|---|---|---|---|
| `.keras` | Keras `save` | Keras | 结构 + 权重 + 训练配置 | PC 保存/继续训练 |
| `.h5` | Keras/TF | Keras | 结构 + 权重 | 旧格式兼容 |
| SavedModel | TensorFlow | TF/TFLite Converter | 图 + 权重 + 签名 | 部署/转换中间格式 |
| `.tflite` | TFLite Converter | TFLite / TFLM | 推理图 + 权重 + 量化参数 | 端侧部署 |
| `model.cc` | `xxd`/工具生成 | C++ 编译器 | `.tflite` 的字节数组 | 编进固件 |
| `.espdl` | esp-ppq | esp-dl | esp-dl 图 + 量化参数 | 另一条路线，不通用 |

关键点：

- `.tflite` 和 C 数组是**同一份字节**，只是打包方式不同；
- `.espdl` 不能给 TFLM 用，`.tflite` 也不能给 esp-dl 用；
- 模型转换不是改后缀，必须经过 Converter。

---

## 12. 周边组件：什么时候用，什么时候不用

| 组件 | 用途 | 你需要吗 |
|---|---|---|
| TensorBoard | 训练可视化 | 训练时可选 |
| tf.data | 数据管道 | 数据大时用 |
| TF Datasets | 现成数据集 | 入门/基准 |
| TF Hub / Kaggle Models | 预训练模型 | 找模型时看 |
| Model Garden | 官方模型实现 | 参考结构 |
| Model Optimization | QAT/剪枝/聚类 | 量化精度不够时 |
| TFLite Converter | 转 `.tflite` | **必须** |
| TFLite Interpreter | PC 验证 | **必须** |
| TFLM / esp-tflite-micro | MCU 推理 | **必须** |
| esp-nn | int8 加速 | **必须**（ESP32） |
| TF Serving | 服务器推理 | 不用 |
| TFX | 生产 ML 流水线 | 不用 |
| TF.js | 浏览器/Node | 不用 |
| TF Probability | 概率模型 | 暂时不用 |
| TF Agents | 强化学习 | 暂时不用 |
| TF Federated | 联邦学习 | 不用 |
| TF Ranking / Recommenders | 排序/推荐 | 不用 |
| TF Graphics / Text / I/O | 专项 | 不用 |

一句话：**训练用 Keras/TF，转换用 TFLite Converter，PC 验证用 Interpreter，板端用 TFLM + esp-nn；其余按需。**

---

## 13. 你的最小栈

### PC 训练侧

| 组件 | 版本/选择 |
|---|---|
| Python | 3.11 |
| TensorFlow | 2.15.1 CPU |
| Keras | 2.15（TF 自带） |
| NumPy | 1.26.x |
| TensorBoard | 可选 |
| tf.data | 可选 |
| Model Optimization | 仅 QAT 时安装 |

### ESP32 侧

| 组件 | 版本/选择 |
|---|---|
| ESP-IDF | 5.5.3 |
| esp-tflite-micro | 1.4.3 |
| esp-nn | 1.5.0 |
| 模型 | int8 `.tflite` / `model.cc` |

### 暂时不用

- TF Serving、TFX、TF.js；
- TF Agents、TF Federated；
- TF Ranking/Recommenders；
- TensorFlow 1.x 的 Graph/Session；
- 完整 TFLite 运行时跑 MCU；
- 上游 `tensorflow/tflite-micro` 直接当组件。

---

## 14. 常见误区

| 误区 | 事实 |
|---|---|
| “装了 TensorFlow 就能在 ESP32 跑” | ESP32 跑的是 TFLM，不是 Python TensorFlow |
| “TFLite 和 TFLite Micro 是一个东西” | 完整 TFLite 和 MCU 版 TFLM 是两套运行时 |
| “Keras 是独立于 TensorFlow 的框架” | Keras 是 API，TensorFlow 是常见后端 |
| “模型改成 `.tflite` 后缀就行” | 必须经过 TFLite Converter |
| “int8 是训练时选 dtype” | int8 是转换阶段产生；QAT 是训练期模拟 |
| “float32 也能用 esp-nn 加速” | esp-nn 主要加速 int8，float32 走参考实现 |
| “esp-tflite-micro 需要 clone 上游 TFLM” | 组件已经 vendored，正常开发不需要 |
| “Hub 上的模型直接能上 MCU” | 要看尺寸、算子、量化、输入、许可证 |
| “模型越小一定越快” | 还取决于算子、内存布局、是否 int8、是否走 esp-nn |

---

## 15. 学习路径

1. **Keras 基础**：Sequential/Functional、Dense/Conv2D、compile/fit；
2. **TensorFlow 工具**：SavedModel、`tf.data`、TensorBoard；
3. **TFLite Converter**：float32 → int8、representative dataset；
4. **TFLite Interpreter**：PC 验证、看输入输出/量化参数；
5. **TFLM**：`GetModel`、OpResolver、arena、Invoke；
6. **esp-tflite-micro + esp-nn**：固件集成、int8 性能；
7. **Model Optimization**：精度不够时再做 QAT；
8. **周边组件**：按项目需要再学，不要一开始全铺开。

---

## 16. 参考

- TensorFlow 官网：<https://www.tensorflow.org/>
- Keras 指南：<https://keras.io/guides/>
- TFLite / LiteRT 文档：<https://ai.google.dev/edge/litert>
- TFLite Micro 文档：<https://ai.google.dev/edge/litert/microcontrollers/overview>
- TensorFlow Model Optimization：<https://www.tensorflow.org/model_optimization>
- TensorBoard：<https://www.tensorflow.org/tensorboard>
- tf.data：<https://www.tensorflow.org/guide/data>
- esp-tflite-micro：<https://github.com/espressif/esp-tflite-micro>
- esp-nn：<https://github.com/espressif/esp-nn>
- 上游 TFLM：<https://github.com/tensorflow/tflite-micro>
- 本仓库实战：[README.md](README.md)
- 公共基础：[model-deployment.md](https://github.com/yh-xh/ESP32-TinyML/blob/main/model-deployment.md)、[neural-networks.md](https://github.com/yh-xh/ESP32-TinyML/blob/main/neural-networks.md)

---

> 本文只梳理关系与边界；具体 API 以对应版本文档为准。
