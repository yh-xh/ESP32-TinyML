# ESP32-P4 / esp-dl 学习笔记

> **适用环境**：ESP-IDF v5.5.3 · esp-dl（本地 checkout）· esp-ppq（Python 量化环境）· ESP32-P4
> **记录时间**：2026-10 ~ 2026-11
> 内容以实测为准，版本更新后部分 API 可能变化，以源码为准。

---

## 目录

- [1. esp-dl 是什么](#1-esp-dl-是什么)
- [2. 模型的基本组成与 esp-dl 的差异（初学者）](#2-模型的基本组成与-esp-dl-的差异初学者)
- [3. 从模型到芯片的完整流程](#3-从模型到芯片的完整流程)
- [4. 模型量化（esp-ppq）](#4-模型量化esp-ppq)
- [5. C++ 端使用 esp-dl](#5-c-端使用-esp-dl)
- [6. 扩展 esp-dl：自定义算子（Module）](#6-扩展-esp-dl自定义算子module)
- [7. 实战踩坑（重点）](#7-实战踩坑重点)
- [8. ESP32-P4 平台配置](#8-esp32-p4-平台配置)
- [9. esp-who 框架速览](#9-esp-who-框架速览)
- [10. 速查命令](#10-速查命令)
- [11. 参考](#11-参考)

---

## 1. esp-dl 是什么

Espressif 官方的端侧深度学习推理框架（C++），在 ESP32-S3 / ESP32-S31 / ESP32-P4 等芯片上做模型推理。

几个关键事实：

- **没有 NPU**：截至 ESP32-P4，推理靠 CPU + RISC-V SIMD（esp-dl 里有 `esp32p4` 专用 ISA/汇编加速路径）。Intel8 卷积的实测有效算力大约 **1 ~ 1.5 GFLOP/s**（P4 @ 360 MHz 双核，粗估）。
- 模型格式为 **`.espdl`**（FlatBuffers），由 PC 端量化工具 **esp-ppq** 生成，运行时用 `fbs_loader` 加载。
- 内置常见任务的预处理/后处理：图像分类、目标检测（yolo11 / espdet / msr / mnp / pico 等）、人脸识别、语音特征。

### 目录结构（`esp-dl/esp-dl/`）

| 目录 | 内容 |
|---|---|
| `dl/base` | 基础类型、ISA 抽象、`dl::TensorBase` |
| `dl/model` | `dl::Model`、`dl_model_base.hpp`（含 `DL_LOG_INFER_LATENCY` 等工具宏） |
| `dl/module` | 算子实现 |
| `dl/tensor`、`dl/math`、`dl/tool` | 张量、数学、工具 |
| `fbs_loader` | `.espdl` 的 FlatBuffers 加载器 |
| `vision/image` | `ImagePreprocessor`、`resize_ppa`、颜色转换、画框工具 |
| `vision/detect` | `Detect` / `DetectImpl` / `DetectWrapper` / 各种后处理器 |
| `vision/classification`、`vision/recognition`、`audio` | 其他任务 |

---

## 2. 模型的基本组成与 esp-dl 的差异（初学者）

> 这一节是概念铺垫：先讲"一个模型到底是什么"，再讲 esp-dl 跑的模型和你在 PC 上训练出来的模型差在哪。
> 有基础的读者可以直接跳到第 3 节。

### 2.1 一个模型 = 结构 + 参数

训练出来的"模型"可以拆成两部分：

1. **结构（计算图 / graph）**：由**算子（operator / layer）**和它们之间的连接组成。
   常见算子：`Conv`（卷积）、`Gemm`/`Linear`（全连接）、`ReLU`/`Sigmoid`（激活）、`MaxPool`（池化）、`Concat`（拼接）、`Reshape`、`Softmax`……
   结构决定"数据怎么流动"：输入 → 算子 1 → 算子 2 → … → 输出。
2. **参数（parameters / weights）**：每个算子内部可学习的数值，主要是**权重 weight** 和**偏置 bias**。
   YOLOv8n 约有 300 万参数，int8 量化后约 3 MB —— 这就是"模型大小"的主要部分。

另外几个必须区分的概念：

- **激活值（activation）**：推理时每层算出来的**中间结果**，不是训练出来的参数，**随输入变化**（换一张图片，每层输出都不同）。量化时"激活的 scale"要靠校准数据统计，原因就在这里。
- **中间张量（intermediate tensor）**：除输入/输出之外的所有张量，是运行时内存占用的主要来源。
- **超参数（hyperparameter）**：不是学出来的，是人定的，例如学习率、输入分辨率 640×640、类别数。
- **算子 vs 层**：日常混用；严格说"层"是结构概念（如一个卷积层），"算子"是计算概念（如 Conv + Bias + ReLU 可能被融合成一个算子）。

### 2.2 推理 vs 训练

| | 训练（training） | 推理（inference） |
|---|---|---|
| 计算方向 | 前向 + 反向传播 | 只有前向 |
| 需要什么 | 梯度、损失函数、优化器、大量数据 | 一张输入图 → 一组输出 |
| 产物 | 权重（`.pth`） | 可部署的模型文件（`.espdl`） |

部署时只保留**前向图**，所以部署文件通常比训练框架的检查点小得多。esp-dl 只做推理，不做训练。

> **💡 打个比方：训练是怎么发生的（射箭）**
>
> 1. **前向传播（射箭）**：你（输入数据）拿起弓，靠当前的姿势和肌肉记忆（**权重**）把箭射出去，箭落在靶子左下角（**输出预测**）。
> 2. **计算误差（Loss）**：教练告诉你，离靶心偏了 10 厘米。
> 3. **反向传播（复盘）**：教练从结果倒推——"手臂抬低了 2°、呼吸没稳住"，算出身体各部位（**各权重**）对偏差应负的"责任大小"（**梯度**）。
> 4. **权重更新**：你按建议修正角度和呼吸节奏（**更新权重**），准备下一次射箭。
>
> **推理就是只做第 1 步**：箭射出去就结束——不算误差、不更新权重。这也解释了为什么部署的模型只需要"前向图"。

### 2.3 张量与内存布局

**张量（tensor）** = 多维数组 + 数据类型 + 形状 + 内存布局。

- **形状**：图像模型常见 `[N, C, H, W]`（batch、通道、高、宽）；**esp-dl 只支持 batch N = 1**。
- **数据类型**：训练用 `float32`；量化后 esp-dl 用 `int8`（权重/激活）和 `int32`（bias 累加）。
- **内存布局**：同一张图，`NCHW` 和 `NHWC` 在内存里的排法不同；做颜色转换、resize 时必须和模型的预期一致。
- **像素格式**也是一种"布局"：摄像头常输出 `RGB565`（2 字节/像素），训练常用 `RGB888`（3 字节）——`ImagePreprocessor` 负责转换。

### 2.4 常见模型文件格式

| 格式 | 里面有什么 | 能直接部署到 esp-dl 吗 |
|---|---|---|
| PyTorch `.pth` | **只有权重**，结构在 Python 代码里 | 不能（需要 PyTorch 解释器） |
| ONNX `.onnx` | 结构 + 权重 + 输入输出签名 + opset，**自包含** | 不能，需先量化成 `.espdl` |
| TensorFlow Lite `.tflite` | 结构 + 权重（自带解释器约定） | 不能，先转 ONNX 再量化 |
| **`.espdl`** | **esp-dl 自己的图 + int8 权重 + 量化参数** | 能 |

> 关键点：`.pth` 和 `.onnx` 的区别不是"是不是权重文件"，而是 **ONNX 自包含**——`.onnx` 的 `graph.initializer` 里就存着训练好的权重（详见 4.5 节）。

### 2.5 浮点模型 → 量化模型

量化就是把 float32 的数值映射到 int8：

```
量化：    q = round(x / scale) + zero_point
反量化：  x ≈ (q - zero_point) * scale
```

- **`scale`** 决定 int8 能表示的数值范围（`scale × 127` 就是上限）。
- **per-tensor**：整个张量共用一个 scale；**per-channel**：每个输出通道一个 scale。esp-dl 的权重多用 per-channel，精度更好。
- **校准（calibration）**：用一批真实输入跑一遍 float 模型，统计每层激活值的范围 → 算出激活的 scale。这就是 `calib_dataloader` 的作用。
- **esp-dl 的特殊表示**：它的 `exponent` 其实就是"scale 用 2 的幂表示"（`scale = 2^exponent`），反量化时可以用移位/查表实现，对没有浮点加速的 MCU 更友好。`.info` 里看到的 `exponents: [-4]` 就是 `scale = 2^-4 = 0.0625`。

### 2.6 通用模型 vs esp-dl 的模型：核心差异

| 维度 | 通用模型（ONNX / PyTorch） | esp-dl 的 `.espdl` |
|---|---|---|
| 文件格式 | Protobuf（ONNX）/ pickle（pth） | **FlatBuffers**（可按偏移直接访问，几乎零解析开销） |
| 算子集合 | 通用、几百个 | esp-dl 自己实现的 Module 子集；**不支持的算子部署不了** |
| 数值类型 | float32 为主 | **int8 权重/激活 + int32 bias + exponent** |
| 输入形状 | 可动态、可多 batch | **固定**（量化时定死），batch 只能是 1 |
| 内存分配 | 框架运行时动态分配 | **静态内存规划**：加载/规划阶段一次性分配，中间张量按生命周期复用 |
| 执行方式 | 解释器逐个算子调度 | **execution plan**：一串 `Module` 按顺序 `forward()`；大算子可选双核拆分 |
| 预处理/后处理 | 图外自己写（Python/C++） | 内置 `ImagePreprocessor` + 检测/分类后处理器 |
| 硬件相关 | 通用 | 和芯片绑定（SIMD/PIE 指令、量化规则），**`.espdl` 不能跨芯片通用** |

一句话概括：**通用模型是"给 PC/GPU 的通用计算描述"，`.espdl` 是"为某一颗 MCU 编译过的、内存和指令都排好的可执行计划"。**

### 2.7 FlatBuffers：既是文件格式，也是内存布局

> `.espdl`（以及 `.tflite`）都基于 **FlatBuffers**。理解它，就能理解为什么 esp-dl 加载模型几乎不需要"解析"。

**传统格式（Protobuf / JSON / pickle）的痛点**

以 `.json` / `.onnx` 为例，数据在"硬盘"和"内存"里是两种形态：

- **硬盘上**：一串按语法排列的字节流。
- **内存里**：程序必须把它读进 RAM，再**解析（parse）**——分配变量、创建对象、字符串转数字，重新搭出一套数据结构。

也就是说：文件是文件，内存对象是内存对象，中间隔着"解析"这道工序。

**FlatBuffers 的做法：让两者变成同一份数据**

FlatBuffers 在序列化时，就按**内存对齐和偏移量**的规则把数据排好：

- 文件里就是一块连续的二进制 Buffer，**它在 flash 里的样子和在 RAM 里的样子完全一致**。
- 运行时不需要"解析成对象树"：把这块 buffer 的基地址交给程序，程序按文件里记录的**偏移量**做指针跳转，就能直接读出任意字段。

> 类比：
> 普通格式（Protobuf）= 乐高积木装在盒子里，玩之前得倒出来按图纸拼好；
> FlatBuffers = 出厂就已经拼好的乐高，打开盒子直接玩，想看哪块看哪块。

**对 MCU 意味着什么**

- **省 RAM**：不需要先解析出一整套对象，也不一定需要把权重重新拷贝一遍。
- **省时间**：没有解析阶段，加载 ≈ 把指针指过去。

**esp-dl 里的实际情况（要分清）**

FlatBuffers 只是"允许"零拷贝，**最终是否真的零拷贝取决于配置**：

- 开 `CONFIG_SPIRAM_RODATA` / `CONFIG_SPIRAM_XIP_FROM_PSRAM` 时，启动日志会提示
  `FbsLoader: ... fbs model is copied to PSRAM` —— 模型被整体拷到 PSRAM（大模型、频繁访问时更稳）。
- 不开时，模型可以留在 flash 的 rodata 里，通过 cache 映射按偏移直接访问。

所以：**"FlatBuffers = 内存布局"的含义是——它的磁盘表示和内存表示是同一份数据；而"要不要把它拷进 RAM"是运行时的策略选择，两件事不要混为一谈。**

### 2.8 一次推理在 esp-dl 里的生命周期

把上面的概念串起来：

```
1. 加载    dl::Model 读 .espdl（FlatBuffers），拿到 module 列表和 int8 权重
2. 规划    minimize() 去掉推理用不到的变量；静态分配中间张量（按生命周期复用）
3. 预处理  ImagePreprocessor：颜色转换 + resize/letterbox → 填进模型输入张量
4. 前向    model.run()：按 execution plan 依次 forward 每个 Module
             ├─ 输入/输出都是 int8 张量
             ├─ 每个张量带自己的 exponent（反量化用）
             └─ 大卷积可选双核拆分（RUNTIME_MODE_MULTI_CORE）
5. 取输出  get_output("名字") → TensorBase（int8 数据 + exponent）
6. 后处理  模型外的 C++ 代码：反量化、解码框、NMS、画到屏幕上
```

### 2.9 给初学者的几个"常识校正"

- **"模型多大"和"运行占多少内存"是两回事**：权重可以放 flash/PSRAM；运行时还要分配激活/中间张量，这部分往往比权重还大（检测模型尤其明显）。
- **推理速度不等于参数量**：yolov8n 参数量不大，但 640×640 的计算量（约 8.7 GFLOPs）才是耗时的原因。
- **量化不是"压缩文件"那么简单**：它改变数值精度，有些算子对量化很敏感（分数、小数值），处理不好会掉点（见 7.1 节）。
- **`model_input` 的 buffer 会被复用**：不要把输入/输出张量的指针长期保存，用完立即取值或拷走。
- **模型文件不能跨芯片通用**：量化时 `target` 不同，量化规则/算子实现都不同，S3 的 `.espdl` 不能丢给 P4 跑。

---

## 3. 从模型到芯片的完整流程

```
训练 (PyTorch / 其他框架)
  → 导出 ONNX（batch=1，onnxsim 简化）
  → esp-ppq 量化（校准数据 + 目标芯片规则）
  → model.espdl（+ .info / .json）
  → 放到 flash rodata / 分区 / SD 卡
  → C++：dl::Model 加载 → ImagePreprocessor → model.run() → 后处理
```

esp-dl 只支持 **batch = 1**，所有环节（导出、量化、推理）都按单张图设计。

---

## 4. 模型量化（esp-ppq）

### 4.1 两个入口

`esp_ppq.api` 提供两个函数，量化后端完全相同，区别只在**图从哪来**：

| 入口 | 输入 | 适用场景 |
|---|---|---|
| `espdl_quantize_torch` | `torch.nn.Module` + 权重（`.pth`） | 有 PyTorch 训练代码；想在 PC 上顺手比量化前后精度 |
| `espdl_quantize_onnx` | `.onnx` 文件 | 模型来自别的框架；**需要在量化前做 ONNX 图手术** |

一个容易忽略的实现细节（`esp_ppq/api/espdl_interface.py`）：

```python
# espdl_quantize_torch 的第一步就是导出 ONNX，然后直接调用 ONNX 版
torch.onnx.export(model, dummy_input, "<name>.onnx", opset_version=...)
return espdl_quantize_onnx(onnx_import_file="<name>.onnx", ...)
```

**所以 torch 版只是"自动导出 ONNX"的便捷封装。** 另外它导出时用的是 `<espdl 文件名>.onnx`，会**覆盖同名 ONNX 文件**，需要保留原始导出的话请换名或先备份。

### 4.2 参数

| 参数 | 说明 |
|---|---|
| `target` | `esp32p4` / `esp32s31` / `esp32s3` / `c`，决定算子支持、量化规则、per-channel 策略 |
| `quant_type` | `w8a8`（默认，权重/激活 8 位）、`w8a16`、`w16a16`、`none`（FP32 导出） |
| `calib_dataloader` | 校准数据 DataLoader |
| `calib_steps` | 校准步数（按 batch 计） |
| `input_shape` | 输入形状，**batch 必须是 1** |
| `collate_fn` | 从 `TensorDataset` 的 `(x, y)` 里只取 `x` |
| `device` | `cpu` / `cuda`（校准在 PC 上跑） |
| `error_report` | 打印逐层/图级量化误差，调参时非常有用 |
| `export_config` | 导出量化配置（`.json`） |
| `export_test_values` | 导出 C++ 对拍用的测试输入/期望输出 |
| `verbose` | 日志详细程度 |

### 4.3 产物

以 `quantize_sin_model` 为例：

| 文件 | 内容 |
|---|---|
| `sin_model.espdl` | 量化后的模型（INT8 权重 per-channel + INT32 bias + exponent） |
| `sin_model.info` | 量化图的文本描述：张量类型/形状/exponent/量化参数 |
| `sin_model.json` | PPQ 配置：`configs`（逐节点量化设置）、`dispatchings`（节点→`ESPDL_INT8`）、`values`（每张量 scale/zero_point + 测试向量） |
| `sin_model.onnx` | 原始 ONNX（torch 入口会生成/覆盖） |

`.info` 里能直接看到量化结果，例如：

```
graph main_graph (
  %input[INT8, 1x1], exponents: [-4]
) initializers (
  %fc1.weight[INT8, 1x1x1x64]     ← 权重被存成 4D、逐通道量化
  %fc2.bias[INT32, 64]            ← bias 量化成 INT32
)
```

### 4.4 校准数据

- `calib_dataloader` **`shuffle` 必须为 `False`**：量化误差分析会多次遍历数据集，顺序变了误差就不可比。
- 校准数据要覆盖真实输入的分布（比如摄像头截帧），不要用纯随机数，否则激活值的 scale 会定得离谱。
- 数据量不用很大：`calib_steps × batch_size` 覆盖分布即可（示例是 32×32）。

### 4.5 ONNX 文件为什么"自带权重"

PyTorch 里结构在代码、权重在 `.pth`；**ONNX 是自包含的**：

- `graph.node` → 算子连接（结构）
- `graph.initializer` → 权重（训练好的张量）
- `graph.input/output` → 输入输出签名
- `opset_import` → 算子版本

实测 `sin_model.onnx`：5 个节点（3×`Gemm` + 2×`Relu`）、6 个 initializer（fc1/fc2/fc3 的 weight+bias），

```text
fc1.weight [64,1]  fc1.bias [64]
fc2.weight [64,64] fc2.bias [64]
fc3.weight [1,64]  fc3.bias [1]
```

与 `sin_model.pth` 逐元素对比全部 `allclose=True`；权重 17,412 B / 文件 18,185 B。

> 大模型（> 2 GB protobuf 上限）会把权重拆到 `model.onnx.data`，`.onnx` 里只存引用，两个文件要一起带。小模型看到孤立的 `.onnx.data` 一般是被 `onnxsim`/重写后内联留下的残留，可以删。

### 4.6 量化后验证

torch 入口可以拿回 PPQ 图，用 `TorchExecutor` 在 PC 上跑**模拟量化推理**，和原模型比 loss：

```python
from esp_ppq.executor.torch import TorchExecutor
executor = TorchExecutor(graph=quant_ppq_graph, device="cpu")
y_quant = executor(batch_x)[0]
```

### 4.7 量化方法全景：PTQ 之外还有什么

先把术语理清：**PTQ（post-training quantization）是"只用校准数据定 scale"的一类基线方法**；TQT、LSQ 等是在 PTQ 之上再做"局部训练"的**优化 pass**，它们不是互斥的"模式"，可以叠加。

**① 数值方案（`quant_type`，4 种 + 1 个调试档）**

| 方案 | 权重 | 激活 | 平台枚举（P4 = v2） | 说明 |
|---|---|---|---|---|
| `w8a8` | int8 | int8 | `ESPDL_INT8` | 默认，最小最快 |
| `w8a16` | int8 | 16 bit | `ESPDL_W8A16` | 8 位激活掉点时优先试 |
| `w16a16` | 16 bit | 16 bit | `ESPDL_INT16` | 精度最高，最慢最占内存 |
| `none` | FP32 | FP32 | — | 只做导出/对照基线 |
| `w16a16` + `debug=True` | — | — | `ESPDL_H_PRE_INT16` | 仅 PC 端高精度模拟，用于判断掉点来源，不部署 |

> 旧的 `num_of_bits=8/16` 参数仍可用，但已被 `quant_type` 取代。

**② 校准算法（`calib_algorithm`）**

| 算法 | 原理 | 默认用在哪 |
|---|---|---|
| `kl` | 直方图 + KL 散度最小化，裁掉长尾 | **int8 激活**（`espdl_setting()` 默认） |
| `percentile` | 按分位数裁剪（默认 **0.9999**） | **int8 权重** |
| `minmax` | 最小/最大值 | **int16** 权重和激活 |
| `mse` | 直方图上最小化量化误差 | 手动选 |
| `isotone` / `constant` / `floating` | 保序 / 常数 / 直接 MSE | 特殊场景，esp-dl 部署基本不用 |

支持按层覆盖：`calib_algorithm_per_layer = {"某层": "minmax"}`；权重和激活可分别设置。

**③ 训练类优化 pass（全部默认关闭，可按需叠加）**

| 方法 | 训练/优化什么 | 数据 | 开关 | 适用场景 |
|---|---|---|---|---|
| **TQT** | 每层裁剪阈值（MSE 目标） | 校准集 | `setting.tqt_optimization` | PTQ 掉点 1~2% 时首选，性价比最高 |
| **LSQ** | 直接训练 step size/scale（STE） | 校准集 | `setting.lsq_optimization` | TQT 不够时；注意 **P4 是 2 的幂量化，LSQ 与约束冲突**，官方建议改用 TQT |
| **AdaRound 类** | 训练权重的取整方向 | 校准集 | 无 setting 快捷开关，需手动加 pass | 权重舍入误差敏感的小模型 |
| **BiasCorrection** | 修正量化后的 bias 偏差 | 校准集 | `setting.bias_correct` | bias 敏感网络 |
| **BlockwiseReconstruction** | 按 block 重建，压逐块输出误差 | 校准集 | `setting.blockwise_reconstruction` | 深层网络误差累积 |
| **Equalization** | 等价变换重分配权重/激活范围 | 校准集 | `equalization_setting` | 通道间动态范围差异大 |
| **完整 QAT** | 权重 + 假量化端到端训练 | **有标签训练集** | `esp_ppq.qat`（无 `.espdl` 导出路径） | esp-dl 链路基本用不上 |

> 另有 `lut_optimization`（默认开）——用查表实现激活函数，属于实现层加速，不改变精度。

**④ 推荐尝试顺序**

```
PTQ（默认 kl/percentile）
  → TQT
  → LSQ / BlockwiseReconstruction
  → BiasCorrection / Equalization
  → 混合精度（w8a16 或个别层 16 bit）
```

所有训练类 setting 都支持 `interested_layers`（只训练指定层，省时间也避免把好的层训练坏）。

> ⚠️ 如果问题是"分数被 Concat 共用量化 scale 压成 0"（7.1 节），换任何量化方法都没用——先做 ONNX 图手术。

### 4.8 自动量化：AutoQuant 与 espdl-quantize Skill

两者**不是两种量化算法**，而是同一套 esp-ppq passes 的两种"搜索驱动器"。

**方式一：AutoQuant（`espdl_auto_quantize_onnx`）**

纯 Python API，在预设搜索空间里枚举/筛选配置：

```python
from esp_ppq.api import AutoQuantSearchSetting, espdl_auto_quantize_onnx

setting = AutoQuantSearchSetting(
    search_mode="fast",          # exhaustive=枚举全部；fast=先筛一轮再细化
    num_of_candidates=5,         # 最终保留 Top-K
    score_direction="maximize",  # 不传 evaluate_fn 时用 "minimize"
    run_dir="outputs/auto_quant",
)
espdl_auto_quantize_onnx(..., evaluate_fn=evaluate_fn, setting=setting)
```

- **评分**：给 `evaluate_fn`（返回 `(score, extras)`，如 top1/mAP）就按它排；不给则用默认的 **graphwise SNR 误差**（最差 3 层平均），此时方向应设 `minimize`。
- **fast 模式**可调 `top_strategy=10`、`early_stop_patience=3`，避免组合爆炸；`strategy_space`/`param_space` 可自定义搜索范围。
- **产出**：`run_dir/` 下 `summary.json`、`candidates.json`，以及 `0000/`、`0001/` 每轮实验的配置和模型文件。
- **仅 ONNX 入口**，确定性、可进 CI。

**方式二：espdl-quantize Skill**

一个给 AI Agent 用的 skill（路径 `esp-dl/tools/agents/skills/espdl-quantize/`），靠"读误差分布 → 选方法 → 再跑一轮"在线迭代：

- 你提供契约 `user_quant.py`：`QUANT_CONFIG`（模型类型/输入形状/目标芯片/primary_metric）、`create_calib_dataloader()`、`get_torch_model()` 或 `onnx_path`、`evaluate(quant_graph)`（以及可选的 `evaluate_fast()`）。
- Agent 自动跑：**Phase 0** baseline → **Phase 1** 读 layerwise 误差/分布找 Top-K 最差层 → **Phase 2** 按决策手册选方法生成 setting → **Phase 3** 执行一轮 → **Phase 4** 比较与收敛判断，循环。
- 内置知识库（`references/ppq_methods.md`）、决策手册（`decision_playbook.md`）、固定 harness（`scripts/run_iteration.py`）和状态机（`compare_iterations.py`）。
- **目标感知安全网**：P4/S3/C 是 2 的幂量化，skill 会**自动跳过 LSQ**（改用直接训练 `log2_scale` 的 TQT）；P4 上启用逐层均衡时给出警告。
- **产出**：每轮 `outputs/iter_N/`（metrics / layerwise_error / setting / 模型），最终 `outputs/final_report.md` + `outputs/best/model.espdl`；支持"再迭代 N 轮"续跑。
- 依赖：pandas/scipy/tqdm；需要 Agent（Claude Code/Cursor/OpenCode 等），上下文越大越稳。

**对比**

| 维度 | AutoQuant | espdl-quantize Skill |
|---|---|---|
| 谁决策 | esp-ppq 搜索算法 | AI Agent + 决策手册/知识库 |
| 搜索方式 | 预设空间枚举/启发式 | 每轮按层误差分布选方向迭代 |
| 输入 | ONNX + 校准数据 + 搜索设置 | 契约文件 + 目标指标/轮数 |
| 模型入口 | 仅 ONNX | Torch / ONNX |
| 反馈信号 | evaluate_fn 或默认 SNR | layerwise 误差/分布 + 指标 + 报告 |
| 运行环境 | 纯 Python，可批处理/CI | 需要 Agent，交互式 |
| 成本 | exhaustive 可能组合爆炸 | 轮数可控，但每轮有 Agent 开销 |
| 产出 | Top-K 候选 | 单一 best + 可读的 final_report |
| 可复现性 | 高（确定性） | 中（依赖 Agent，但每轮 setting 落盘） |
| 额外依赖 | 无 | pandas/scipy/tqdm |

**怎么选**

- 刚上手 / 批量模型 / 无 Agent 环境 → **AutoQuant**（先 `fast`）。
- 掉点卡住，想知道"为什么、哪层最差、试过哪些方案" → **Skill**（报告和决策手册就是干这个的）。
- 单个小模型、时间紧 → 手动 `espdl_quantize_onnx` + TQT 跑两轮通常就够，不必上自动搜索。

**在 Claude Code 里接入 skill（本机实测）**

```bash
# 方式 1（需要 Node.js）：npx skills add espressif/esp-dl --skill espdl-quantize
# 方式 2（无需 Node）：复制整个 skill 目录到 Agent 的 skills 目录
cp -r esp-dl/tools/agents/skills/espdl-quantize ~/.claude/skills/espdl-quantize

# 依赖装进【跑 harness 的那个 Python 环境】（esp_ppq 所在环境）
pip install -r ~/.claude/skills/espdl-quantize/assets/extra_requirements.txt
```

装好后在 Claude Code 会话里直接说"用 espdl-quantize skill 调一下某个模型"，或输入 `/espdl-quantize`。注意 harness 必须用装了 `esp_ppq` 的解释器运行。

> skill 只能调量化参数，修不了图结构问题（如 7.1 节的 Concat 共 scale）。

---

## 5. C++ 端使用 esp-dl

### 5.1 模型放哪里

esp-dl 的模型加载支持三种位置（由组件 Kconfig 选择，函数签名里的 `location` 参数与之对应）：

| 位置 | 说明 |
|---|---|
| Flash rodata | 模型直接编进固件（`_binary_xxx_espdl_start`），启动即可用，占固件空间 |
| Flash 分区 | 烧到独立分区，可用 `esptool` 单独更新模型 |
| SD 卡 | 从文件系统读，方便换模型，需要挂载 SD |

开启 `SPIRAM_RODATA` / `SPIRAM_XIP_FROM_PSRAM` 时，加载日志会提示 **fbs 模型被拷贝到 PSRAM**（大模型推荐）。

### 5.2 加载与推理（示意）

```cpp
#include "dl_model_base.hpp"
#include "dl_image_preprocessor.hpp"

// 1) 加载
dl::Model model(path, /*name=*/0, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
model.minimize();   // 删除无用推理变量（注意：会破坏调试/测试，见日志警告）

// 2) 预处理：mean/std + letterbox（参数必须与训练一致）
dl::image::ImagePreprocessor pre(&model, {0, 0, 0}, {255, 255, 255});
pre.enable_letterbox({114, 114, 114});

dl::image::img_t img{data, (uint16_t)w, (uint16_t)h,
                     dl::image::DL_IMAGE_PIX_TYPE_RGB565LE};
ESP_ERROR_CHECK(pre.preprocess(img));

// 3) 推理
model.run();

// 4) 取输出（名字来自量化产物的 .info）
dl::TensorBase *out = model.get_output("/model.22/Sigmoid_output_0");
```

要点：

- **mean/std 必须和训练一致**。只做 `x/255` 的模型用 `{0,0,0}` / `{255,255,255}`，不要照抄 ImageNet 的归一化。
- 自定义检测模型时，`dl::detect::DetectImpl` 的 `run()` 流程是：
  `preprocess → model->run() → postprocess() → get_result(w, h)`。
  `get_result()` 只做 `limit_box()/limit_keypoint()` 裁剪，不做坐标映射。
- 后处理器里的坐标反映射用 `ImagePreprocessor` 提供的量：
  `get_resize_scale_x(true)` / `get_resize_scale_y(true)`（`true` = 取逆缩放）、`get_border_left()` / `get_border_top()`。
- 张量反量化：`float v = dl::dequantize(int8_value, DL_SCALE(tensor->exponent))`。

### 5.3 lazy_load

`DetectWrapper` 的子类（如官方 `HumanFaceDetect`）默认参数是 **`lazy_load = true`**：模型推迟到第一次 `run()` 才加载。**官方所有 example 调用时都传 `false`**（在 `app_main` 里、任务启动前加载），大模型尤其应该这样，避免和摄像头/显示流水线并发加载。

---

## 6. 扩展 esp-dl：自定义算子（Module）

esp-dl 的算子以 `dl::module::Module` 子类的形式存在。新增一个算子 =
**C++ 侧实现 Module + 注册 + 量化侧支持 + 对齐验证**。

> ⚠️ 官方文档「如何创建新模块（算子）」的示例接口是旧版：文档里写的是
> `forward(std::vector<TensorBase*>&, ...)`，而当前 esp-dl 的基类是
> `forward(ModelContext*, runtime_mode_t)`。照抄文档会编译失败，以下以本地源码为准。

### 6.1 需要重写的接口

| 方法 | 作用 |
|---|---|
| `get_output_shape()` | 由输入形状推输出形状 |
| `forward(ModelContext*, runtime_mode_t)` | 高级接口：实际计算，张量从 `context->get_tensor(m_inputs_index[i])` 取 |
| `forward_args(void*)` | 低级接口：按参数执行，可留空 |
| `deserialize(fbs_model, node_name)` | 从 `.espdl` 的节点信息重建模块 |
| `print()` | 调试输出 |

### 6.2 最小骨架（与本地版本对齐）

```cpp
namespace dl {
namespace module {

class MyCustomModule : public Module {
public:
    MyCustomModule(const char *name = "MyCustomModule",
                   module_inplace_t inplace = MODULE_NON_INPLACE,
                   quant_type_t quant_type = QUANT_TYPE_NONE)
        : Module(name, inplace, quant_type) {}

    std::vector<std::vector<int>> get_output_shape(std::vector<std::vector<int>> &input_shapes) override
    {
        return {input_shapes[0]};      // 按算子语义计算
    }

    void forward(ModelContext *context, runtime_mode_t mode) override
    {
        TensorBase *input  = context->get_tensor(m_inputs_index[0]);
        TensorBase *output = context->get_tensor(m_outputs_index[0]);
        if (quant_type == QUANT_TYPE_SYMM_8BIT) {
            // int8 实现（部署必需）
        } else if (quant_type == QUANT_TYPE_FLOAT32) {
            float *in  = (float *)input->get_element_ptr();
            float *out = (float *)output->get_element_ptr();
            // ...
        }
    }

    static Module *deserialize(fbs::FbsModel *fbs_model, std::string node_name)
    {
        quant_type_t q;
        fbs_model->get_operation_attribute(node_name, "quant_type", q);
        return new MyCustomModule(node_name.c_str(), MODULE_NON_INPLACE, q);
    }

    void print() override { ESP_LOGI("MyCustomModule", "%s", name.c_str()); }
};

} // namespace module
} // namespace dl
```

### 6.3 注册

在 `esp-dl/dl/module/include/dl_module_creator.hpp` 的 `register_dl_modules()` 里加一行：

```cpp
this->register_module("MyCustomModule", MyCustomModule::deserialize);
```

键名是 **ONNX op type 字符串**（现成的有 `"Conv"`、`"Add"`、`"Mul"`……），必须和 `.espdl`
里记录的一致，否则 `ModuleCreator::create()` 找不到它，模型加载直接失败。

### 6.4 五个容易踩的坑

1. **`quant_type` 不是装饰**：真实模块在 `forward` 里按 `QUANT_TYPE_SYMM_8BIT` /
   `QUANT_TYPE_SYMM_16BIT` / `QUANT_TYPE_FLOAT32` 分支实现（参考 `dl_module_relu.hpp`）。
   只写 `QUANT_TYPE_NONE` 无法部署。
2. **`inplace` 影响静态内存规划**：能原地计算的算子标 `MODULE_INPLACE_CHANGED_BUFFER`
   可以省一块缓冲；标错会读到已被覆盖的数据。
3. **属性名要和 PPQ 写的一致**：`deserialize` 里 `fbs_model->get_operation_attribute(...)`
   读的键（如 `quant_type`）必须与 esp-ppq 导出时写入的一致——这就是文档说的"接口应与 ONNX 对齐"。
4. **量化侧也要"认识"这个算子**：`.espdl` 是 esp-ppq 从 ONNX 图导出的；C++ 侧新增的算子
   如果在 ONNX/PPQ 里没有对应表示，量化时根本不会生成它。
5. **多核**：`forward` 会收到 `runtime_mode_t`，大算子可以用 `module_forward_dual_core`
   拆分（参考 `dl_module_conv.hpp`）。

### 6.5 验证

- `model->test()`：用 `export_test_values=True` 导出的测试向量在板端对拍（容差读模型
  metadata `test_max_error`，默认 5e-4）；
- `profile(true)`：按层打印耗时/内存；
- 可以对照 esp-dl 自带模块（`dl/module/include/`，例如 `dl_module_relu.hpp`）的写法。

### 6.6 espdl-operator Skill（AI Agent 辅助）

`esp-dl/tools/agents/skills/espdl-operator/` 提供算子开发的完整流程指导：

| 文件 | 内容 |
|---|---|
| `SKILL.md` | 研究 → 实现（C++ / C 参考核 / SIMD）→ 量化支持（esp-ppq）→ 测试 → 优化 → 文档 |
| `references/esp-dl-templates.md` | Module 模板 |
| `references/esp-ppq-checklist.md` | 量化侧支持清单 |
| `assets/docker/` | 构建/测试用 Docker 镜像（基于 `espressif/idf:v5.4.3`） |

接入 Claude Code（与 espdl-quantize 相同）：

```bash
cp -r esp-dl/tools/agents/skills/espdl-operator ~/.claude/skills/espdl-operator
```

> 注意：该 skill 的测试阶段（Phase 6）依赖 **Docker**（镜像 `espdl/idf-ppq:latest`，
> 首次构建约 20~30 分钟）。没装 Docker 时，实现/量化阶段仍可用，测试需要改用本地
> ESP-IDF 工具链。

---

## 7. 实战踩坑（重点）

### 7.1 朴素 YOLOv8 导出会让分数全归零

**症状**：检测框能出来，但 score 全是 0（或全部低于阈值）。

**原因**：ONNX 导出时 box 和 score 被 `Concat` 拼成一个输出，量化时两者**共用一个 scale**；score 的真值范围（0~1）被 box 的数值范围（0~640）拖累，int8 分辨率不够，分数被截断成 0。

**解决**：在 ONNX 图上做手术，把 `Concat` 拆成两个独立输出（如 `box [1,4,8400]`、`score [1,3,8400]`），让它们各自量化。后处理器按两个张量分别解码。

**经验**：量化前先看 `.info` 里各输出的 `exponent`/scale。如果某个张量的可表示范围明显小于它需要的范围（例如 score 只能表示到 0.248），就是 scale 被别的张量拖累了。

### 7.2 P4 没有 NPU：耗时估算与输入尺寸

ESP32-P4 实测（yolov8n，int8）：

| 输入 | 实测/估算耗时 | anchors |
|---|---|---|
| 640×640 | **5.0 ~ 6.7 s** | 8400 |
| 480×480 | ~3.0 s | 4725 |
| 384×384 | ~2.1 s | 3024 |
| 320×320 | **~1.4 s** | 2100 |

esp-dl 内置耗时日志（`esp-dl/dl/dl_define.hpp` 里 `#define DL_LOG_INFER_LATENCY 1`）会打印：

```
detect: pre:   35 ms      ← 预处理
detect: model: 5.0~6.7 s  ← 推理（99%）
detect: post:  22~46 ms   ← 后处理
```

**耗时几乎全在推理**，预处理/后处理可以忽略。缩小输入是唯一有效的提速手段；换输入尺寸后记得改后处理器里的 `num_anchors`（8400 / 4725 / 3024 / 2100）。

### 7.3 `cam_fb_peek()` 的 use-after-free（esp-who）

**崩溃指纹**：

```
Guru Meditation Error: Core 0 panic'ed (Load/Store access fault)
MEPC 在 lv_memcpy / lv_draw_sw_*（LVGL 画 canvas）
MTVAL ≈ 0x4FFC0000            ← 内部 SRAM 的最高地址
```

**原因**：`WhoFrameCapNode::cam_fb_peek()` 返回 ringbuf 里的**裸 `Frame*`**，而节点任务在轮转 ringbuf 或 `cleanup()` 时会 `delete fb`。显示端（`WhoFrameLCDDisp`）在锁外使用这个指针，可能读到**已被释放、内存又被别的分配复用**的结构体——实测拿到过一个 640×480 的解码节点帧，而 canvas 是 1024×600，LVGL 按 1024×600 去读写 → 越界。

**解决**：节点锁内把帧结构体拷贝出来（`cam_fb_copy`），并在显示端校验 `width/height` 与节点一致、对不上就丢帧。像素缓冲本身不会被释放（PPA 节点用自己的静态缓冲），所以拷贝是安全的；但**解码节点的 `cleanup()` 会 `heap_caps_free(fb->data)`**，那种节点不能只拷贝。

> 识别例子 `WhoRecognitionAppLCD` 给 `WhoFrameLCDDisp` 传 `peek_index = 1` 是对同一问题的部分规避（index 0 是最先被删的那一帧），不是设计意图。

### 7.4 慢模型 + Yield2Idle：一个框都画不出来

`WhoDetectAppLCD` + 秒级推理的模型会出现"**画面正常但永远没有框**"，换快模型（~0.1 s）就正常——**快推理把下面两个坑掩盖了**：

1. **画框被 `is_active()` 挡掉**。`lcd_disp_cb()` 用 `m_task->is_active()` 做门，而 `is_active()` 只要 `TASK_PAUSE` 位被置上就返回 false。`WhoYield2Idle` 每 `(WDT_TIMEOUT - MAX_TASK_LOOP_TIME)/2` 秒给从不 idle 的核置一次暂停位；慢模型要等当前推理结束（可达数秒）才应答 → 这段窗口内每帧直接 return。
   **改法**：只在 `TASK_STOPPED/TASK_STOP` 时跳过。画框只依赖当前结果，跟检测任务跑不跑无关。
2. **暂停时的 `cleanup()` 会清掉结果**。`cleanup_for_paused()` → `WhoDetect::cleanup()` → `WhoDetectResultLCDDisp::cleanup()` 清 `m_results` + `m_result`。而检测任务的暂停应答**恰好发生在推理结束、结果刚存进来的那一刻** → 结果在显示任务看到之前就被清掉（实测 `save: items=1` 之后 44 ms 的 draw 就是 `queue=0 items=0`）。
   **改法**：暂停时什么都不清，只在真正 stop 时清。

### 7.5 显示与检测的时间戳对齐

`WhoDetectResultLCDDisp::lcd_disp_cb()` 只画"时间戳不晚于当前显示帧"的结果：

```cpp
if (displayed_fb.timestamp >= result.timestamp) { 显示 } else { 等待 }
```

上游的设计是**让显示端看 ringbuf 里较旧的一帧**（识别例子 `peek_index = 1`），使"画面"和"检测结果对应的帧"对齐。这个把戏只在推理延迟 < 显示延迟（≈ 一两帧）时成立；推理几秒的模型，框必然落后画面几秒——**唯一解是加快推理**（缩小输入/换小模型）。

### 7.6 分数标注：字段有，但上游不填

`draw_item_t` 有 `label` 字段，`draw_items_on_canvas()` 也会渲染它，但上游的 `save_detect_result()` 从来不赋值 → 框上一直没有文字。自己填：

```cpp
char label[24];
snprintf(label, sizeof(label), "%.2f", res.score);
item.label = label;
```

注意框坐标可能是**负的**（模型常给出 `[102,-6,1024,403]`），标签起点要夹进画布，否则文字画到屏幕外。

### 7.7 PPA 缩放的 1/16 量化规则

esp-dl 的 `resize_ppa()` 把缩放比存成 `int + frac/16`，实际写出的尺寸是
`in.block_w * scale_int + in.block_w * scale_frac / 16`，不足的部分留右/下黑边。

所以 **crop 尺寸要落在 1/16 网格上**。例如把 640×480 裁成 630×369 再放到 1024×600：

```
1024/630 = 1.6260 → 1 + 10/16 = 1.625
 600/369 = 1.6260 → 1 + 10/16 = 1.625
输出：1023×599（只有 1 像素黑边）
```

换成 640×375（scale 1.6 → 1.5625）就会留 24 列 × 15 行黑边。

### 7.8 USB UVC 摄像头的坑（esp-who）

1. **`usb_host_uvc` 2.4.2 会把 `urb_size` 压到摄像头的 `dwMaxPayloadTransferSize`**（`uvc_host.c:838`），配置的 10240 会塌成 3072 = 一个 ISOC 包 → 满屏 `frame error` / `missed EoF`。该 clamp 在 **2.5.0** 移除。BSP 5.2.3 的依赖链锁在 `esp_video 2.0.1` → `usb_host_uvc 2.4.*`，只能本地 patch `managed_components`（重新解析依赖会丢）。
2. **`VideoCapture` 没有设置 `VIDIOC_S_DQBUF_TIMEOUT`**，继承 esp_video 的 `portMAX_DELAY`：UVC 真卡住时 `DQBUF` 会永久阻塞，画面冻结。可以自己设 1 s 超时。
3. 残留的 `usb err 1` / `missed EoF`（~0.3/s）与供电、温度无关，属正常范围。

### 7.9 Phantom Kconfig symbol

IDF 5.5.3 + 组件管理器：依赖规则引用了**本地没有任何组件定义的 Kconfig 符号**时，`prepare.py` 会直接退出（`Missing required kconfig option after retry`）。

解决：在 `main/Kconfig.projbuild` 里声明它，**必须带 prompt 字符串**：

```kconfig
config ESP_VIDEO_USE_CUSTOMIZED_ESP_H264_VERSION
    bool "Use customized esp_h264 version (workaround, keep disabled)"
    default n
```

无提示的隐藏符号在值为 `n` 时会被 kconfiglib 省略，组件管理器读不到，等于没定义。

### 7.10 依赖版本要钉住

组件管理器会把 `~` 版本浮动到最新。实测踩过：

- `esp_lvgl_port` 必须钉 `2.8.0~1`（2.9.0 的 `esp_lvgl_port_disp.c` 用了一个 IDF 5.5.3 不存在的字段 `on_frame_buf_complete`）
- `lvgl` 钉 `9.5.0`
- 用 `dependencies.lock.<BSP>` 记录能跑的组合

另外 `CONFIG_LV_MEM_SIZE_KILOBYTES` / `CONFIG_LV_ASSERT_HANDLER_INCLUDE` 会以坏值（0 / 空字符串）残留在 `sdkconfig` 里，表现为编译期 `#error`；把这些值显式写进 BSP 的 sdkconfig 默认文件并**删除 sdkconfig** 重配。

---

## 8. ESP32-P4 平台配置

### 8.1 芯片版本（v1.3）

IDF 5.5 对 esp32p4 的默认最低芯片版本是 v3.0，v1.3 的板子必须在 `sdkconfig.defaults` 里放开：

```kconfig
CONFIG_ESP32P4_SELECTS_REV_LESS_V3=y
CONFIG_ESP32P4_REV_MIN_0=y
```

不写会报 `Target doesn't support chip revision`。代价：**flash 频率被压到 40 MHz**（烧录命令行里能看到 `--flash_freq 40m`）。

### 8.2 Flash 与启动布局

- 板载 16 MB，`CONFIG_ESPTOOLPY_FLASHSIZE_16MB=y`
- P4 的烧录偏移：bootloader `0x2000`、分区表 `0x8000`、app `0x10000`（与 S3 的 `0x0/0x8000/0x10000` 不同）

### 8.3 控制台

ESP32-P4 默认 **UART0 + USB-Serial-JTAG 双输出**（`CONFIG_ESP_CONSOLE_UART_DEFAULT` + `CONFIG_ESP_CONSOLE_SECONDARY_USB_SERIAL_JTAG`）。`/dev/ttyACM0` 能直接看到日志。

### 8.4 ESP32-S3 → ESP32-P4 迁移清单

1. `idf.py set-target esp32p4`（旧的 `build/` 最好先删；半截目录 `fullclean` 会拒绝清理）
2. **SPI 主机名兼容**：P4 有三个 SPI 外设，`SPI1_HOST=0 / SPI2_HOST=1 / SPI3_HOST=2`，S3 代码里的 `SPI2_HOST` 可以直接用
3. **GPIO 范围**：P4 为 GPIO0 ~ GPIO54；具体引脚要和板载外设（屏、摄像头、SD、I2C）核对
4. **组件化**：把驱动拆到 `components/<name>/`，`idf_component_register(SRCS ... INCLUDE_DIRS include REQUIRES ...)`。
   **头文件里暴露了 `gpio_num_t` / `spi_host_device_t` 时，依赖必须是公开的 `REQUIRES`**（`PRIV_REQUIRES` 只在组件内部可见，main 会找不到头文件）
5. 如果板子有 BSP，把版本相关的 sdkconfig 写进 BSP 的默认配置文件，而不是手改 `sdkconfig`

---

## 9. esp-who 框架速览

### 9.1 数据管线

```
摄像头 (MIPI-CSI 或 USB-UVC)
  → WhoFetchNode      (取原始帧，ringbuf)
  → WhoDecodeNode     (JPEG → RGB565；UVC 路径需要)
  → WhoPPAResizeNode  (PPA 硬件缩放/裁剪 → 屏幕尺寸)
  → 显示任务 (WhoFrameLCDDisp：LVGL canvas 指向 fb，零拷贝)
       └→ 检测任务 (WhoDetect：peek 同一节点，跑模型，结果回调)
            └→ WhoDetectResultLCDDisp：把框画进 canvas 的 layer
```

### 9.2 关键类

| 类 | 作用 |
|---|---|
| `WhoFrameCap` / `WhoFrameCapNode` | 管线容器 / 节点基类（每个节点一个 FreeRTOS 任务 + ringbuf） |
| `WhoDetect` | 检测任务：`cam_fb_peek → model->run() → 结果回调` |
| `WhoDetectAppBase` / `WhoDetectAppLCD` | 组装采集+检测+显示+画框 |
| `WhoFrameLCDDisp` | 把 fb 设为 LVGL canvas 缓冲并刷新（可选 `peek_index`） |
| `WhoDetectResultLCDDisp` | 保存检测结果，在 `lcd_disp_cb` 里画框/文字 |
| `WhoYield2Idle` | 定期暂停任务让 CPU 进 idle（**慢模型下是坑的来源**，见 7.4） |

### 9.3 线程与核心

- 帧管线节点、LCD 显示任务：核心 0，优先级 2
- 检测任务：核心 1，优先级 2
- 视频/显示与检测是**解耦**的，检测慢不会直接卡住画面，但会因 Yield2Idle 的暂停机制互相影响

---

## 10. 速查命令

```bash
# 编译 / 烧录 / 监视
idf.py build
idf.py -p /dev/ttyACM0 flash monitor

# 新建/切换目标
idf.py set-target esp32p4

# 打开 esp-dl 推理耗时日志
#   esp-dl/esp-dl/dl/dl_define.hpp:  #define DL_LOG_INFER_LATENCY 1

# 用当前 ELF 解析 panic 地址
riscv32-esp-elf-addr2line -pfiaC -e build/<app>.elf 0x481b9588 0x4807ef42

# 量化示例
python quantize_onnx_model.py     # 或 quantize_torch_model.py
```

---

## 11. 参考

- esp-dl 仓库：<https://github.com/espressif/esp-dl>
- esp-ppq（量化工具）：<https://github.com/espressif/esp-ppq>
- esp-who：<https://github.com/espressif/esp-who>
- ESP-IDF 编程指南：<https://docs.espressif.com/projects/esp-idf/>
- 量化教程示例：`esp-dl/examples/tutorial/how_to_quantize_model/`（sin_model / mobilenetv2 / yolo11n …）

---

> 本文记录来自实际项目调试，欢迎提 Issue 交流或补充。
