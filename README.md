# TFLite Micro + esp-nn 学习笔记

> **适用环境**：ESP-IDF v5.5.2（C5）/ v5.5.3（S3）· esp-tflite-micro 1.3.4 / 1.4.3 · esp-nn 1.2.5 / 1.5.0 · ESP32-C5 / XIAO ESP32S3 Sense
> **记录时间**：2026-11
> 本分支由原来的 ESP32-C5 手势识别笔记和 XIAO ESP32-S3 人形检测笔记合并而来。两条笔记原本各自按项目组织，这里按**一条完整链路**重新整理：数据 → 重训练 → 转换/量化 → 模型进固件 → 运行时 → 实测验证。
> 公共基础见主分支 [model-deployment.md](https://github.com/yh-xh/ESP32-TinyML/blob/main/model-deployment.md)；另一条路线见 [esp-dl + esp-ppq 分支](https://github.com/yh-xh/ESP32-TinyML/tree/esp-dl-esp-ppq)。

> **两个案例的分工**
> - **ESP32-C5 手势识别**：IMU 时序数据、3 分类、float32 模型。它把一个完整的“采集 → 训练 → TFLM 部署 → 滑窗/去抖”流程走通，同时也是一个反面教材：**float32 模型不会走 esp-nn 的 int8 加速内核**。
> - **XIAO ESP32-S3 人形检测**：OV2640 摄像头、2 分类、int8 模型。它展示了 int8 量化、ESP-NN 加速、摄像头预处理和可复现的实测数据。
>
> 两个案例共用同一条 TFLM + esp-nn 链路。先看 C5 理解“链路怎么走”，再看 S3 理解“怎么把性能和量化做对”。

---

## 目录

- [1. 路线与两个案例](#1-路线与两个案例)
- [2. 端到端链路总览](#2-端到端链路总览)
- [3. 阶段一：问题定义与数据](#3-阶段一问题定义与数据)
- [4. 阶段二：训练](#4-阶段二训练)
- [5. 阶段三：转换与量化](#5-阶段三转换与量化)
- [6. 阶段四：模型进入固件](#6-阶段四模型进入固件)
- [7. 阶段五：固件运行时](#7-阶段五固件运行时)
- [8. 阶段六：实测验证](#8-阶段六实测验证)
- [9. 两个案例的横向对比](#9-两个案例的横向对比)
- [10. 踩坑清单](#10-踩坑清单)
- [11. 速查与关键数字](#11-速查与关键数字)
- [12. 参考](#12-参考)

---

## 1. 路线与两个案例

### 1.1 TFLite Micro + esp-nn 的分工

**TFLite Micro（TFLM）是运行时 + 模型格式**，esp-nn 是它下面的 **int8 算子实现库**。把两者分开看，很多问题会变得清楚：

| 层 | 负责什么 | 不负责什么 |
|---|---|---|
| `.tflite` 模型 | 图结构、权重、量化参数 | 不负责调度和内存分配 |
| TFLM | 解析 FlatBuffers、分配 `tensor_arena`、按顺序调用算子 | 不提供高性能卷积实现 |
| esp-nn | 卷积、深度卷积、池化、全连接、softmax 的 int8 优化 kernel | 不解析模型，不管理张量生命周期 |
| 你的应用 | 数据采集、预处理、后处理、任务/阈值/去抖 | 不实现算子 |

这条路线适合：

- 已经有 TFLite 模型，或者团队更熟悉 TensorFlow/Keras；
- 模型可以量化到 int8，且目标芯片有 esp-nn 优化（S3/P4 最好，C5 走通用优化）；
- 需要跨平台、快速验证，而不是深度定制 Espressif 的 `.espdl` 格式。

如果做人脸检测/识别并想直接用 esp-dl 的检测框、关键点、识别后处理，走 [esp-dl + esp-ppq](https://github.com/yh-xh/ESP32-TinyML/tree/esp-dl-esp-ppq) 更省事。

### 1.2 两个案例

| 项 | ESP32-C5 手势识别 | XIAO ESP32S3 Sense 人形检测 |
|---|---|---|
| 任务 | 3 分类：画圈 / V / Unknown | 2 分类：person / no-person |
| 传感器 | BMI270，200 Hz，3 轴角速度 | OV2640，96×96 灰度 |
| 输入 | 200 帧 × 3 轴，float32 | 96×96×1，int8 |
| 模型大小 | 11,072 字节 | 300,568 字节 |
| 量化 | float32 | 全 int8 |
| IDF / 运行时 | v5.5.2，TFLM 1.3.4 + esp-nn 1.2.5 | v5.5.3，TFLM 1.4.3 + esp-nn 1.5.0 |
| 芯片 | ESP32-C5，240 MHz，软浮点 | ESP32-S3，240 MHz |
| arena | 50 KB | 请求 160 KB，实用约 118 KB |
| 实测重点 | 功能、滑窗、去抖 | 取帧/推理耗时、帧率、arena |
| 关键结论 | 链路能跑通；float32 不走 esp-nn | int8 + ESP-NN 把推理压到约 50 ms |

### 1.3 先记住三个“必须一致”

无论换芯片、换模型还是重训练，下面三件事只要有一件不一致，端侧结果就会错，而且往往不会报错：

| 必须一致 | 训练侧 | 端侧 |
|---|---|---|
| 输入形状/布局 | `(200, 3)` 或 `(96, 96, 1)` | `memcpy`/reshape/摄像头输出的顺序 |
| 标签顺序 | `0=圈, 1=V, 2=Unknown` 或 `0=no-person, 1=person` | `kLabels[]`、`kPersonIndex` 等索引 |
| 预处理/量化 | 归一化方式、输入 scale/zero_point | 输入转换、输出反量化 |

后文每个阶段都会回到这三条。先把它们记住，比记住某个 API 更有用。

---

## 2. 端到端链路总览

### 2.1 一张图

```
问题定义
  → 数据采集（传感器/摄像头）
  → 清洗、划分、校准集
  → PC 训练（Keras/PyTorch）
  → TFLite Converter（float32 或 int8 量化）
  → .tflite 模型
  → C 数组 / EMBED_FILES / 分区 / 文件系统
  → 固件：GetModel → OpResolver → Interpreter + arena
  → 预处理（与训练一致）
  → Invoke
  → 后处理（反量化、argmax、阈值、去抖）
  → 实测验证（耗时、内存、准确率）
```

### 2.2 每一阶段的产物与验收标准

| 阶段 | 产物 | 验收标准 |
|---|---|---|
| 问题定义 | 类别表、输入规格 | 标签顺序固定；输入尺寸/类型确定 |
| 数据采集 | 原始数据 + 标签 | 覆盖真实场景；训练/测试按人/场次划分 |
| 训练 | Keras/PyTorch 模型 | 测试集准确率达标；混淆矩阵可接受 |
| 转换 | `.tflite` | 输入输出 shape/dtype/量化参数正确 |
| 模型进固件 | C 数组或模型分区 | 在 flash/RAM 的位置明确；符号/长度正确 |
| 运行时 | 可推理固件 | `AllocateTensors()` 成功；输出合理 |
| 预处理/后处理 | 端到端结果 | 与训练侧预处理一致；阈值/去抖合理 |
| 实测 | 耗时/内存/准确率 | 可复现；瓶颈明确 |

### 2.3 两个关键决策

1. **float32 还是 int8？**
   - float32：转换简单、精度通常略好，但 esp-nn 只加速 int8，float32 走 TFLM 参考实现。
   - int8：模型更小、更快，能用 esp-nn；需要代表性校准集，精度可能略降。
   - C5 手势模型很小，float32 够用；S3 图像模型较大，必须 int8 + ESP-NN 才能实时。

2. **模型放哪里？**
   - 先选 C 数组/`EMBED_FILES`，简单、启动即可用；
   - 需要单独更新模型时，再考虑独立分区或文件系统。详见第 6 章。

---

## 3. 阶段一：问题定义与数据

### 3.1 先定义问题和标签

| 项 | C5 手势 | S3 人形检测 |
|---|---|---|
| 输入 | 200 帧 × 3 轴角速度 | 96×96 灰度图 |
| 类别 | 0=画圈，1=V，2=Unknown | 0=no-person，1=person |
| 正/负类 | 三个动作 + 兜底 Unknown | 有人/无人 |
| 输出 | 3 个 float 概率 | 2 个 int8 分数 |
| 业务阈值 | 置信度 < 85% 忽略 | person score > 60% 视为有人（显示路径） |

标签顺序是硬约束。训练时用 `0/1/2`，端侧也必须是同样的顺序；不要只改标签文字而不改索引。

### 3.2 C5：把动作变成 CSV

C5 用 BMI270 采集角速度：

- 陀螺仪 200 Hz、±2000 dps；每 5 ms 一帧，200 帧 = 1 秒。
- 每帧 3 个数（gx, gy, gz），一行 CSV 共 600 个数。
- COLLECT 模式下，角速度绝对值和超过阈值（默认 100 dps）才开始录 200 帧；INFERENCE 模式持续采样 + 滑窗推理。

数据采集的要点：

- **Unknown 类必须采**：各种“不是目标手势”的动作、静止、随手晃动。没有 Unknown，模型会在圈/V 之间硬选。
- **换人、换方向、换速度**：泛化主要靠数据多样性，不靠网络结构。
- **日志会污染 CSV**：串口同时输出 `I (...) ...` 和纯数字行，训练前必须过滤，只保留数字/逗号行。
- **采样节拍有抖动**：`vTaskDelay(5ms)` 不是硬件定时器，BMI270 轮询也可能偶尔没准备好；训练和推理要用同一套采集方式。

### 3.3 S3：从摄像头到 96×96 灰度

S3 的 person_detection 模型要求 96×96 单通道灰度：

- 摄像头配置为 `FRAMESIZE_96X96` + `PIXFORMAT_GRAYSCALE`，直接出灰度，省掉 RGB 转换。
- 取帧后把 uint8 灰度逐字节转成 int8：`image_data[i] = fb->buf[i] ^ 0x80`。
- 如果是自定义的人脸/非人脸模型，仍然建议统一到 96×96 灰度；采集时就用同样的分辨率/裁剪方式，避免训练和端侧视角不一致。

数据采集方式：

- 复用摄像头 Web demo 的 `/camera` 返回 JPEG，或把 JPEG 存到 SD 卡再拷到 PC；
- PC 端统一 `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` + `cv2.resize(img, (96, 96))`；
- 正类：不同人、光照、角度、距离、遮挡；负类：背景、天空、墙面、物体、模糊画面、部分人脸；
- 负类要覆盖真实场景，否则“像人脸的背景”会被误判成正类。

### 3.4 清洗、划分、校准集

| 事项 | C5 | S3 |
|---|---|---|
| 清洗 | 过滤 `I/W/E` 日志行；保持每行 600 个数 | 去掉模糊/重复帧；统一灰度/尺寸 |
| 划分 | 按采集场次/人划分，不能随机打乱相邻窗口 | 按人/场景/时间划分，不能随机打乱相邻帧 |
| 校准集 | 不需要（float32） | 需要，100–500 张真实画面，覆盖光照/背景/人距 |
| 数据泄漏 | 同一段动作的相邻滑窗不能跨训练/测试 | 同一段视频的相邻帧不能跨训练/测试 |

数据泄漏是最常见的“测试准确率很高、实机却不行”的原因。按场次/人划分比随机划分更可信。

### 3.5 数据阶段检查清单

- 输入尺寸、通道数、数据类型和端侧完全一致；
- 标签顺序固定，类别数量与输出层一致；
- 负类/Unknown 足够多，覆盖真实场景；
- 训练/测试按人/场次/时间划分；
- 量化模型留有独立校准集，且校准集不参与训练。

### 3.6 两个案例的硬件入口

**C5 手势（Sensairshuttle / ESP32-C5）**：

| 信号 | GPIO | 说明 |
|---|---:|---|
| I2C SCL | 3 | BMI270 时钟 |
| I2C SDA | 2 | BMI270 数据 |
| SDO | 9 | 地址选择，拉低 → `0x68` |
| BOOT 按键 | 28 | 采集/模式切换 |

IMU 配置：陀螺仪 200 Hz、±2000 dps；加速度计 200 Hz、±2g（推理不用加速度）。

**S3 人形检测（XIAO ESP32S3 Sense / OV2640）**：

| 信号 | GPIO | 说明 |
|---|---:|---|
| XCLK | 10 | 20 MHz |
| SIOD / SIOC | 40 / 39 | SCCB |
| VSYNC / HREF / PCLK | 38 / 47 / 13 | 同步与像素时钟 |
| D0–D7 | 15, 17, 18, 16, 14, 12, 11, 48 | Y2–Y9 |
| PWDN / RESET | -1 / -1 | 不控制 |

平台配置：8 MB flash、8 MB Octal PSRAM（80 MHz）、240 MHz、USB Serial/JTAG；摄像头 96×96 灰度，`fb_count=2`，帧缓冲放 PSRAM。`main/Kconfig.projbuild` 的自定义引脚 `range` 上限要是 48，否则 GPIO48 选不到。

---

## 4. 阶段二：训练

### 4.1 C5：一个小 1D CNN

C5 的手势模型是一个很小的 1D CNN，输入 `(200, 3)`，输出 3 类。结构如下：

| 层 | 输出形状 | 参数量 | 作用 |
|---|---:|---:|---|
| 输入 | 200×3 | 0 | 1 秒、3 轴角速度 |
| Conv1D(8, k=5) | 200×8 | 128 | 找局部波形 |
| MaxPool(4) | 50×8 | 0 | 降采样，容忍速度差异 |
| Conv1D(16, k=5) | 50×16 | 656 | 组合动作片段 |
| MaxPool(4) | 12×16 | 0 | 再降采样 |
| GlobalAvgPool | 16 | 0 | 压成特征向量 |
| Dense(32) | 32 | 544 | 特征组合 |
| Dropout(0.2) | 32 | 0 | 防过拟合 |
| Dense(3, softmax) | 3 | 99 | 三类概率 |

总计约 **1,427 个参数**，float32 权重约 5.7 KB。训练配置：Adam(1e-3)、`sparse_categorical_crossentropy`、300 epochs、batch 64、70/30 划分。

注意：`Conv1D`/`MaxPooling1D` 转到 TFLite 后可能变成 `Conv2D` + `Reshape`/`ExpandDims`、`MaxPool2D`、`Mean`。端侧注册算子时要以实际 `.tflite` 为准，不能只看 Keras 层名。

### 4.2 S3：官方模型与自定义模型

S3 的案例直接用官方 `person_detection` 预训练 int8 模型：

- 输入 96×96×1，输出 2 类，标签 `notperson`/`person`；
- 模型 300,568 字节，5 个算子；
- 它不是人脸识别，只是“整图里有没有人”的二分类。

如果要换成人脸/非人脸：

- 输入仍是 96×96 灰度，输出仍是 2 类；
- 数据换成 face / non-face，标签顺序固定；
- 模型结构可以更小（小型 CNN 或 MobileNetV1 0.25）；
- 训练完走第 5 章的 int8 转换和第 6 章的替换流程。

### 4.3 训练阶段要记录的“模型卡”

重训练最容易丢的信息不是权重，而是这些约定。建议在训练脚本旁边固定记录：

| 项 | 示例 |
|---|---|
| 输入 shape / dtype | `(200,3)` float32；`(96,96,1)` int8 |
| 标签顺序 | `0=圈, 1=V, 2=Unknown` |
| 预处理 | 归一化到 `[0,1]` 或 `[-1,1]`；是否减均值 |
| 输出 | 3 类 softmax；2 类 softmax |
| 量化 | float32 / int8；输入输出 scale/zero_point |
| 模型文件 | 文件名、大小、生成脚本 |

端侧的所有常量（`kNumCols`、`kNumRows`、`kCategoryCount`、`kLabels[]`、`kPersonIndex`）都要从这里来。

### 4.4 什么时候需要重训练

| 触发 | 要改什么 |
|---|---|
| 换传感器/安装位置 | 数据分布变了，重采数据、重训 |
| 增加/删除类别 | 输出层、标签、端侧索引、阈值 |
| 改输入尺寸/窗口长度 | 模型输入、端侧 `kNumCols/kNumRows` 或 `NUM_SAMPLES` |
| 量化后精度下降太多 | 校准集、量化方法、必要时混合精度 |
| 实机误报/漏报高 | 补负类/困难样本，重新划分数据 |

**只改端侧常量、不重训模型，是最常见的错误。** 输入形状变了但模型没变，轻则结果乱，重则内存越界。

---

## 5. 阶段三：转换与量化

### 5.1 float32 还是 int8

| | float32 | int8 |
|---|---|---|
| 转换难度 | 低，不需要校准集 | 中，需要 representative dataset |
| 模型大小 | 4 字节/权重 | 1 字节/权重 |
| esp-nn | **不走**优化内核 | 走 `esp_nn_*_s8` |
| 速度 | 慢（C5 还是软浮点） | 快 |
| 精度 | 通常略好 | 可能略降，需验证 |
| 适合 | 小模型、低频、原型 | 较大模型、实时、省内存 |

C5 模型只有 1,427 个参数，float32 也能跑，所以选择了最简路线；S3 的图像模型 300 KB，必须 int8 + ESP-NN 才能到约 50 ms。

### 5.2 C5：float32 转换

C5 的转换很直接，不做量化：

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_model = converter.convert()
```

产物 **11,072 字节**，输入 `(1,200,3)` float32，输出 `(1,3)` float32。

关键结论：**float32 模型即使链接了 esp-nn，也不会走 esp-nn 的优化 kernel**。C5 笔记里专门验证过这一点：`esp_nn_conv_s8` 只在 int8 分支被调用，float32 走 `reference_ops::Conv`。

### 5.3 S3：int8 全整数量化

S3 的模型是 int8，输入输出也都是 int8。转换的核心是 `representative_dataset`：

```python
def representative_dataset():
    for image in calib_images:          # 100~500 张真实灰度图
        x = image.astype(np.float32) / 255.0
        yield [x.reshape(1, 96, 96, 1)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
```

产物 300,568 字节；输入 `[1,96,96,1]` int8，输出 `[1,2]` int8；实测输出 `scale=0.00390625`（1/256）、`zero_point=-128`。

### 5.4 转换后必须检查的 6 件事

| 检查项 | 怎么查 | 不对会怎样 |
|---|---|---|
| 输入 shape | `interpreter.get_input_details()` | 端侧写入越界或只填一部分 |
| 输入 dtype | 同上 | 用 `data.f` 读 int8，结果乱飞 |
| 输入 scale/zero_point | 同上 | 预处理映射错误，准确率骤降 |
| 输出 shape/dtype | `get_output_details()` | 索引错、反量化错 |
| 输出 scale/zero_point | 同上 | 概率全错 |
| 实际算子列表 | `tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)` 或 Netron | 端侧 `Didn't find op` |

C5 和 S3 的算子列表：

| 项目 | 算子 |
|---|---|
| C5 手势 | Softmax, FullyConnected, MaxPool2D, Conv2D, AveragePool2D, ExpandDims, Reshape, Mean |
| S3 人形检测 | AveragePool2D, Conv2D, DepthwiseConv2D, Reshape, Softmax |

端侧 `MicroMutableOpResolver<N>` 的 `N` 至少等于实际算子数；少一个就会在 `AllocateTensors()`/`Invoke()` 报错。

### 5.5 量化常见问题

- **校准集不代表真实分布**：只用训练集或几张室内图，量化后实机精度会掉。校准集要覆盖真实光照、背景、人距。
- **输出 scale 被别的张量拖累**：多输出模型要分别看每个输出的 scale；某个输出范围明显不够，可能是量化时共用了 scale。
- **不要假设输入映射固定是 `^0x80`**：`^0x80` 只是 uint8→int8 的位映射，真实数值由 `input->params.scale/zero_point` 决定；自定义模型必须按参数重算。
- **量化后要重新测准确率**：不要只看模型大小和速度；用混淆矩阵对比量化前后，确认误报/漏报可接受。

---

## 6. 阶段四：模型进入固件

### 6.1 三种常见方式

| 方式 | 做法 | 优点 | 缺点 | 适用 |
|---|---|---|---|---|
| C 数组 | `xxd -i` 生成 `.cc`，编进固件 | 最简单，启动即可用 | 换模型要重编/重烧 | 大多数固定模型 |
| `EMBED_FILES` | CMake 直接嵌入 `.tflite` | 不生成巨大 C 文件，编译快 | 仍跟固件一起烧 | 模型较大、想避免 C 源文件 |
| 独立分区 / SD | 运行时读取/mmap | 模型/app 分离，可单独更新 | 要加载逻辑、对齐、生命周期 | 需要 OTA/换模型 |

### 6.2 C 数组：`xxd -i` 到底做了什么

```bash
xxd -i model.tflite > model.cpp
```

它把 `.tflite` 的每个字节写成 `0xAB,` 形式的 C 数组，不做模型转换。生成后必须手动补两件事：

```c
alignas(8) const unsigned char model_tflite[] = {
    0x1c, 0x00, 0x00, 0x00, ...
};
const unsigned int model_tflite_len = 11072;
```

- **`const`**：让数组进 flash 的 `.rodata`；没有 `const` 会进 `.data`，上电复制到 RAM。C5 模型 11 KB 还好，S3 模型 300 KB 就会白占 300 KB RAM。
- **`alignas(8)`**：FlatBuffers 的 8 字节对齐要求。
- **符号名**：`xxd` 默认按文件名生成（`model.tflite` → `model_tflite`）；路径带目录会变成下划线。名字和代码里的 `extern` 不一致会链接失败。
- **长度类型**：`xxd` 生成 `unsigned int`，官方头文件可能声明 `const int`；两边要统一。

C5 的模型 11,072 字节，C 源文件约 70 KB；S3 的模型 300,568 字节，C 源文件约 1.9 MB。后者编译明显更慢，模型再大就建议用 `EMBED_FILES`。

### 6.3 怎么确认模型真的在 flash

不要只看 `nm` 的符号类型。ESP32-S3 的 flash 数据映射也在 `0x3C...` 段，`nm` 可能显示 `D`，但不代表内部 RAM。用 section 判断：

```bash
xtensa-esp32s3-elf-readelf -sW build/person_detection.elf | grep g_person_detect_model_data
xtensa-esp32s3-elf-readelf -SW build/person_detection.elf | grep flash.rodata
```

S3 实测：符号在 `.flash.rodata`，大小 `0x49618`（300,568 字节）；`idf.py size` 里 `Flash Data 379,000`、`.rodata 378,744`，模型占了大头。

S3 的烧录布局和大小（本项目）：

| 项 | 值 |
|---|---:|
| bootloader | `0x0` |
| partition table | `0xc000` |
| ota_data_initial | `0x13000` |
| app | `0x20000` |
| app 分区大小 | `0x1E0000`（1.875 MB） |
| 固件大小 | 约 759 KB（`0xb94f0`），余量 61% |

分区表偏移是 `0xc000`，不是常见的 `0x8000`；烧录命令里的偏移要一起改。

### 6.4 外置模型：对齐与生命周期

如果模型放独立分区或 SD：

- **缓冲区必须对齐**，至少 8 字节；FlatBuffers 对 64 位字段有对齐要求。
- **模型缓冲区在推理期间不能释放/复用**；`tflite::GetModel()` 只拿指针，不拷贝。
- **校验长度和 schema 版本**；读取长度和 `model_len` 不一致会解析异常。
- **SD/文件系统要先完整读入连续内存**，不能把 `FILE*` 直接传给 `GetModel`。
- 分区方案可以用 `esp_partition_mmap` 直接映射，避免一次大拷贝；SD 方案通常要读进 PSRAM。

模型只有十几 KB（C5）时，C 数组最省事；模型 300 KB（S3）且需要单独更新时，再考虑分区/SD。

---

## 7. 阶段五：固件运行时

### 7.1 初始化顺序

TFLM 的初始化逻辑是固定的：

```
GetModel(模型指针)
  → 检查 schema version
  → 注册算子（MicroMutableOpResolver）
  → 创建 MicroInterpreter（传 model、resolver、arena、arena_size）
  → AllocateTensors()
  → 取 input/output 张量指针
```

实际代码骨架（不是伪代码，API 就是这些；算子以 C5 的 8 个为例）：

```c
const tflite::Model *model = tflite::GetModel(model_data);
if (model->version() != TFLITE_SCHEMA_VERSION) {
    return;
}

static tflite::MicroMutableOpResolver<8> resolver;
resolver.AddSoftmax();
resolver.AddFullyConnected();
resolver.AddMaxPool2D();
resolver.AddConv2D();
resolver.AddAveragePool2D();
resolver.AddExpandDims();
resolver.AddReshape();
resolver.AddMean();

static tflite::MicroInterpreter interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
if (interpreter.AllocateTensors() != kTfLiteOk) {
    return;
}

TfLiteTensor *input = interpreter.input(0);
TfLiteTensor *output = interpreter.output(0);
```

S3 只注册 5 个算子（AveragePool2D、Conv2D、DepthwiseConv2D、Reshape、Softmax），把上面的 `AddXxx()` 换成模型实际需要的即可。推理时填充输入、调用 `Invoke()`、读取输出，具体见 7.5 和 7.6。

### 7.2 算子注册

C5 和 S3 的算子数量不同：

| 项目 | 算子数 | 说明 |
|---|---:|---|
| C5 手势 | 8 | 1D 层转成 2D 后用到 MaxPool2D/AveragePool2D/Mean 等 |
| S3 人形检测 | 5 | AveragePool2D, Conv2D, DepthwiseConv2D, Reshape, Softmax |

`MicroMutableOpResolver<N>` 的 `N` 是容量，不是“必须正好等于”。少注册一个，运行时会报：

```
Didn't find op for builtin opcode 'CONV_2D' version '3'
```

换模型后，先用 `Analyzer` 或 Netron 看实际算子，再补 `AddXxx()`；不要照抄旧项目的算子列表。

### 7.3 Tensor arena

| 项目 | arena | 实测/说明 |
|---|---:|---|
| C5 手势 | 50 KB | 静态数组，放内部 RAM；未单独记录 `arena_used_bytes()` |
| S3 人形检测 | 100 KB + 60 KB ESP-NN scratch = 163,840 | 实测使用 120,996 字节 |

S3 的分配策略是“先内部 RAM，失败再 PSRAM”：

```c
tensor_arena = heap_caps_malloc(kTensorArenaSize,
                                MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
if (tensor_arena == NULL) {
    tensor_arena = heap_caps_malloc(kTensorArenaSize,
                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}
```

要点：

- `AllocateTensors()` 失败时，先打印 `interpreter.arena_used_bytes()`，再按实际用量加 10–20% 余量。
- PSRAM 更大但更慢；arena 放 PSRAM 会拉长推理时间。
- ESP-NN 的 scratch 是 arena 的一部分（S3 代码里额外加 60 KB），关掉 ESP-NN 后可以减掉。
- 不要盲目设 1 MB：内部 RAM 还要留给摄像头、Wi-Fi、任务栈。

### 7.4 esp-nn 什么时候真的生效

这是整条路线最容易被误解的地方：

| 模型类型 | 卷积走哪 | esp-nn 是否加速 |
|---|---|---|
| int8 | `esp_nn_conv_s8` | 是 |
| float32 | `tflite::reference_ops::Conv` | 否 |

C5 是 float32，所以虽然固件里链接了 esp-nn，推理热路径并没有走优化内核；S3 是 int8，才真正用上 ESP-NN。

确认方法：

1. **看代码分支**：int8 才进 `esp_nn_conv_s8`；
2. **看符号**：`nm build/xxx.elf | grep esp_nn` 只能证明链接了，不能证明执行了；
3. **看耗时**：在 `Invoke()` 前后打点，对比 int8 和 float32 模型；
4. **看 profiling**：接 TFLM 的 `MicroProfiler`。

S3 的实测约 49.6 ms，是 int8 + ESP-NN 的结果；S3 的 `nm` 能看到 `esp_nn_conv_s8_esp32s3`、`esp_nn_conv_s8_3x3_opt` 等 S3 专用符号。ESP32-C5 不在汇编/SIMD 专用列表里，走通用优化 C，所以即使 int8，加速幅度也不如 S3。

ESP-NN 相关的配置：

```
CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN=y
CONFIG_NN_OPTIMIZED=y
CONFIG_NN_OPTIMIZATIONS=1
```

代码里的 `scratchBufSize` 条件编译用的是 `CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN`，不是 `CONFIG_NN_OPTIMIZED`；两个都要对。

### 7.5 预处理：训练和端侧必须同构

**C5**：

- 输入是 200 帧 × 3 轴 float32；端侧 `memcpy` 到 `input->data.f`，顺序必须是 `(200,3)`，不能写成 `(3,200)`。
- 训练时若减均值/归一化，端侧也要做同样的处理；C5 当前是原始角速度值，训练和推理一致。

**S3**：

- 摄像头输出 96×96 uint8 灰度，端侧 `^0x80` 映射到 int8；输入量化参数以模型为准。
- 如果自定义模型输入 scale/zero_point 不同，不能照抄 `^0x80`，要按 `input->params` 重算。
- 训练时的缩放/裁剪/灰度化方式要和端侧一致；训练用 RGB、端侧用灰度，准确率通常会对不上。

### 7.6 后处理

| 项目 | 后处理 |
|---|---|
| C5 | 3 类 argmax + 置信度阈值 85% + 连续 3 次确认、5 次后重置 + 30 帧滑窗（150 ms） |
| S3 | 2 类反量化 + 取较大者/60% 阈值；串口打印 person/no-person 百分比 |

C5 的去抖状态机是业务逻辑，不属于模型：

- 置信度 < 85%：忽略；
- 同类连续 3 次：打印一次；
- 同类连续 5 次：重置，准备识别下一个动作；
- 类别变化：重新计数。

S3 的显示路径用 60% 作为“有人”的门槛，是为了减少误报；二分类也可以直接 argmax（等价 50%），阈值按业务调。

### 7.7 任务结构

| 项目 | 任务结构 | 关键点 |
|---|---|---|
| C5 | `collect`（5 ms 采样）+ `inference`（滑窗推理） | 队列传指针 + 信号量；推理前先 `memcpy`，采样任务只等复制那一瞬间 |
| S3 | 单个 `tf_main` 循环：取帧 → 推理 → 后处理 | 8 KB 栈；`vTaskDelay(1)` 让出 CPU/喂狗 |

什么时候拆任务：

- 采集有严格节拍（IMU 200 Hz）→ 采集和推理分开，避免推理阻塞采样；
- 摄像头 + 推理单线程足够（S3 取帧 0.5 ms、推理 50 ms）→ 单循环更简单；
- 如果需要同时推流/显示，再把推理放到独立任务，注意 PSRAM/CPU 竞争。

---

## 8. 阶段六：实测验证

### 8.1 测量方法

不要用一个端到端帧率代表所有环节。至少分开测：

| 环节 | 怎么测 | 说明 |
|---|---|---|
| 取数/预处理 | `esp_timer_get_time()` 包住采集/转换 | S3 的 `GetImage()` |
| `Invoke()` | 包住 `interpreter.Invoke()` | 模型真实计算时间 |
| 后处理 | 包住反量化/argmax/去抖 | 通常很小，但别忽略 |
| 循环帧率 | 统计每秒完成次数 | 包含日志、delay、任务切换 |
| arena | `arena_used_bytes()` | 真实张量需求 |
| 模型位置 | `readelf`/`idf.py size` | flash 还是 RAM |

测量时把每帧日志改成每秒一次，否则串口输出本身会拖慢帧率。S3 的 16 fps 是**带每帧日志**的结果，纯推理上限约 20 fps。

### 8.2 C5 功能验证

C5 的验证重点是“链路和业务逻辑正确”，不是极限性能：

- 模型 1,427 参数，11,072 字节 float32；arena 50 KB；
- 滑窗 30 帧 = 150 ms 一次推理；
- 串口实际输出：

```
I (9992) Model Pipeline: Motion detected: Counterclockwise circle (confidence: 100.0%)
I (13232) Model Pipeline: Motion detected: V (confidence: 99.9%)
```

这说明采集、训练、转换、TFLM 推理、argmax、去抖整条链路是通的。

**注意**：C5 笔记没有记录单独的 `Invoke()` 耗时。模型很小，通常几毫秒到十几毫秒，但这是量级判断，不是实测数字；要写进性能报告，按 8.1 的方法在 C5 上单独打点。另外，C5 是 float32 + 软浮点，esp-nn 不在热路径上，不要把它当成 int8 加速的例子。

### 8.3 S3 实测结果

S3 的数据是可复现的实测：

| 项 | 实测值 |
|---|---:|
| 取帧 + 转 int8 | 0.52 ms |
| `Invoke()` | **49.6 ms** |
| 循环帧率（含每帧串口日志 + 1 ms delay） | **16 fps** |
| 纯推理理论上限（1000 / 49.6） | 约 20 fps |
| 模型大小 | 300,568 字节 |
| arena 请求 / 实用 | 163,840 / 120,996 字节 |
| 输入 / 输出 | 9,216 / 2 字节 |
| 输出量化 | scale=1/256，zero_point=-128 |
| app 固件 | 约 759 KB（0xb94f0） |
| app 分区余量 | 61% |

串口节选：

```
INFO arena=120996/163840 input=9216 output=2 out_scale=3906/1e6 out_zp=-128
PERF capture=0.523ms invoke=49.601ms loop=16fps
person score:13%, no person score 87%
```

无人场景下 person score 在 9%–14% 波动；把镜头对准人时 person score 会明显升高。这个 demo 不联网，推理不依赖 Wi-Fi。

### 8.4 结果怎么读

- **取帧 0.52 ms vs 推理 49.6 ms**：瓶颈在推理，不在摄像头。不要再去降摄像头分辨率，先看模型/量化。
- **循环 16 fps vs 纯推理 20 fps**：差值是每帧串口打印和 1 ms delay；做性能测试时先关日志。
- **arena 120,996 / 163,840**：还有余量，但不要盲目缩小；换模型后重新测。
- **模型在 flash**：300 KB 不占 RAM；如果误放 `.data`，内部 RAM 会非常紧张。
- **int8 + ESP-NN**：49.6 ms 是这个组合的结果；float32 或关掉 ESP-NN 会明显变慢。自己对比时，保持同一模型、同一输入、同一日志频率。

### 8.5 精度验证

性能之外，必须单独验证准确率：

| 项目 | 验证内容 |
|---|---|
| C5 | 混淆矩阵；Unknown 被误判成手势的比例；按场次/人划分测试集 |
| S3 | person/no-person 的误报/漏报；无人场景分数分布；有人场景阈值 |
| 人脸/非人脸 | 正负类平衡；不同光照/角度/遮挡；量化前后对比 |
| 重训练后 | 固定测试集回归；同一批图片/动作对比旧模型 |

只看总准确率不够。门禁类应用更怕误报，唤醒类应用可能更怕漏报；阈值要按业务调。

### 8.6 验证清单

- 输入 shape/dtype/量化参数与模型一致；
- 标签顺序与训练一致；
- 实际算子全部注册；
- `AllocateTensors()` 成功，`arena_used_bytes()` 在预算内；
- 输出反量化正确，分数范围合理；
- 单独测过 `Invoke()` 和循环帧率；
- 量化前后/重训练前后准确率有对比；
- 实机跑了正类、负类、边界场景。

---

## 9. 两个案例的横向对比

| 维度 | ESP32-C5 手势 | XIAO ESP32-S3 人形检测 |
|---|---|---|
| 传感器 | BMI270，200 Hz，3 轴 | OV2640，96×96 灰度 |
| 输入 | 200×3 float32 | 96×96×1 int8 |
| 模型 | 1D CNN，1,427 参数，11,072 B | 官方 person_detect，300,568 B |
| 量化 | float32 | int8 |
| esp-nn | 链接了但热路径不走 | 真正走 int8 内核 |
| 平台 | ESP32-C5，RISC-V，软浮点 | ESP32-S3，Xtensa，240 MHz |
| arena | 50 KB | 160 KB 请求，约 118 KB 实用 |
| 推理间隔 | 150 ms 滑窗 | 每次取帧后立即推理 |
| 实测性能 | 功能验证；未单独记录 Invoke | 49.6 ms/次，16 fps（含日志） |
| 主要价值 | 展示完整链路和业务去抖 | 展示 int8/ESP-NN 的性能 |

### 9.1 为什么 C5 用 float32、S3 用 int8

- C5 模型只有 1,427 个参数，float32 的模型和计算量都很小；C5 没有 FPU，但模型小到软浮点也能接受。
- S3 的模型 300 KB，float32 会更大更慢；图像任务要求实时，int8 + ESP-NN 是必选项。
- **模型小 → float32 可以先跑通；模型大/要求实时 → 直接按 int8 设计**，不要等 float32 跑不动再回头改。

### 9.2 什么时候选这条路线

- 已有 TFLite 模型、需要快速部署；
- 能接受 int8 量化，目标芯片是 S3/P4（esp-nn 优化最好）；
- 任务是分类/回归，不需要 esp-dl 的检测框/关键点后处理。

如果要做人脸检测/识别、需要现成的框/关键点/embedding 后处理，选 [esp-dl + esp-ppq](https://github.com/yh-xh/ESP32-TinyML/tree/esp-dl-esp-ppq) 更合适。二分类“人脸/非人脸”仍然可以走本文路线，但“识别是谁”不是二分类，需要 embedding + 相似度。

---

## 10. 踩坑清单

### 10.1 数据/训练

| 现象 | 原因 | 处理 |
|---|---|---|
| 改了窗口/输入尺寸，结果乱飞 | 模型输入固定，端侧按新尺寸拷贝 | 重训模型，同步改 `NUM_SAMPLES`/`kNumCols` |
| 类别打印反了 | 标签顺序/索引不一致 | 训练标签、`kLabels[]`、端侧索引三者对齐 |
| 测试准确率高，实机差 | 随机划分导致相邻帧泄漏 | 按人/场次/时间划分 |
| CSV 读入报错 | 串口日志混进数据 | 过滤 `I/W/E` 行，只保留数字/逗号 |
| 同样动作有时识别有时不识别 | 采样节拍抖动、训练/推理采集方式不同 | 统一采集方式；必要时用硬件定时器/DRDY |

### 10.2 转换/量化

| 现象 | 原因 | 处理 |
|---|---|---|
| 模型能跑但分数乱 | 输入 dtype/量化参数不匹配 | 打印 `input->type/scale/zero_point`，按参数转换 |
| 输出永远 50% | 输出反量化错/索引错 | 用 `data.int8` + `params.scale/zero_point` |
| 量化后精度骤降 | 校准集不代表真实分布 | 重做 representative dataset，覆盖光照/背景/人距 |
| 以为 float32 用了 esp-nn | float32 走参考实现 | 想用 esp-nn 必须转 int8 |
| 某个输出分数全 0 | 多输出共用 scale，范围被拖累 | 拆输出、分别量化，检查 `.info`/输出 scale |

### 10.3 模型/固件

| 现象 | 原因 | 处理 |
|---|---|---|
| `Didn't find op ...` | 算子没注册/容量不够 | 按实际模型补 `AddXxx()`，调大 `N` |
| `AllocateTensors() failed` | arena 太小 | 看 `arena_used_bytes()`，加 10–20% |
| 模型占 300 KB RAM | `xxd` 生成非 `const` 数组 | 加 `alignas(8) const`，确认在 `.rodata` |
| 链接失败/符号找不到 | `xxd` 变量名和 `extern` 不一致 | 用 `-n` 固定名字，或同步改头文件 |
| 模型外置后跑飞 | 缓冲区未对齐/提前释放 | 8 字节对齐，保持到推理结束，校验长度 |
| 误以为 `nm` 的 `D` 是 RAM | flash DROM 也在 `0x3C...` | 用 `readelf` 看 section |

### 10.4 运行时/任务

| 现象 | 原因 | 处理 |
|---|---|---|
| 启动后 panic/栈溢出 | 任务栈太小 | S3 `tf_main` 用 8 KB；大局部数组改静态/堆 |
| 采样被推理阻塞 | 采集和推理同一任务 | 拆分任务，队列传指针，先拷贝再推理 |
| 切模式后按键自己触发 | 信号量残留 | 切模式时清空信号量 |
| 初始化失败但程序继续跑 | 错误没传播 | 初始化返回 `esp_err_t`，失败停止创建任务 |
| 帧率比预期低 | 每帧串口打印 | 日志改成每秒一次，再测 |

### 10.5 硬件/平台

| 现象 | 原因 | 处理 |
|---|---|---|
| 摄像头初始化失败 | 引脚/XCLK/PWDN 配置错 | 对照 XIAO 引脚表；XCLK 20 MHz；PWDN/RESET=-1 |
| GPIO48 选不到 | Kconfig `range` 上限 39 | 改 `main/Kconfig.projbuild` 为 48，`idf.py reconfigure` |
| PSRAM 分配失败 | 没开 Octal PSRAM/模式选错 | 开 `CONFIG_SPIRAM_MODE_OCT`、80 MHz |
| 烧录时 `/dev/ttyACM0` 消失 | VMware USB 重新枚举失败 | 关 monitor、重插、`esptool --before=no_reset` |
| SD 挂载后 LED 不亮 | GPIO21 同时是 SD CS 和 LED | 不同时用；或改硬件 R11/R12 |
| 把 person 检测当人脸识别 | 模型只做整图二分类 | 人脸/非人脸可换模型；身份识别用 embedding |

---

## 11. 速查与关键数字

### 11.1 命令

```bash
# 环境与构建
. $IDF_PATH/export.sh
idf.py set-target esp32s3        # 或 esp32c5
idf.py build
idf.py -p /dev/ttyACM0 flash

# 大小与模型位置
idf.py size
nm build/person_detection.elf | grep esp_nn
xtensa-esp32s3-elf-readelf -sW build/person_detection.elf | grep g_person_detect_model_data

# 生成模型 C 数组
xxd -i -n model_tflite model.tflite > model.cpp

# 手动烧录（ttyACM0 掉线/已在下载模式）
python -m esptool --chip esp32s3 -p /dev/ttyACM0 -b 460800 \
  --before=no_reset --after=hard_reset write_flash \
  --flash_mode dio --flash_freq 80m --flash_size 8MB \
  0x0 build/bootloader/bootloader.bin \
  0xc000 build/partition_table/partition-table.bin \
  0x13000 build/ota_data_initial.bin \
  0x20000 build/person_detection.bin
```

### 11.2 关键数字

| 项 | C5 手势 | S3 人形检测 |
|---|---:|---:|
| 输入 | 200×3 float32 | 96×96×1 int8 |
| 模型 | 11,072 B | 300,568 B |
| 参数量 | 约 1,427 | —（官方模型） |
| 量化 | float32 | int8 |
| arena | 50 KB | 160 KB 请求 / 约 118 KB 实用 |
| 推理 | 未单独记录 | 49.6 ms |
| 循环/推理节拍 | 150 ms/次滑窗 | 16 fps（含日志） |
| 平台 | ESP32-C5 240 MHz | ESP32-S3 240 MHz |

### 11.3 从重训练到部署的一页检查

1. 固定输入 shape/layout、标签顺序、预处理；
2. 按人/场次划分数据，留校准集；
3. 训练，记录模型卡；
4. 选 float32 或 int8；int8 做 representative dataset；
5. 转换后检查输入/输出 shape、dtype、scale/zero_point、算子；
6. 生成 C 数组或 `EMBED_FILES`，加 `const`/`alignas(8)`；
7. 注册实际算子，分配 arena，`AllocateTensors()`；
8. 端侧预处理/反量化与训练一致；
9. 单独测 capture/invoke/loop，确认瓶颈；
10. 用固定测试集验证量化前后/重训练前后的准确率。

---

## 12. 参考

- esp-tflite-micro：<https://github.com/espressif/esp-tflite-micro>
- esp-nn：<https://github.com/espressif/esp-nn>
- esp32-camera：<https://github.com/espressif/esp32-camera>
- TFLite Micro 文档：<https://ai.google.dev/edge/litert/microcontrollers/overview>
- TFLite Converter 文档：<https://ai.google.dev/edge/litert/models/tflite_converter>
- ESP-IDF 编程指南：<https://docs.espressif.com/projects/esp-idf/>
- XIAO ESP32S3 Sense 入门：<https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/>
- 公共基础：[model-deployment.md](https://github.com/yh-xh/ESP32-TinyML/blob/main/model-deployment.md)、[neural-networks.md](https://github.com/yh-xh/ESP32-TinyML/blob/main/neural-networks.md)
- 另一条路线：[esp-dl + esp-ppq](https://github.com/yh-xh/ESP32-TinyML/tree/esp-dl-esp-ppq)

---

> 本文由 ESP32-C5 手势识别和 XIAO ESP32S3 Sense 人形检测两个实际项目合并整理；欢迎提 Issue 交流或补充。
