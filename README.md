# ESP32-C5 / esp-nn 学习笔记

> **适用环境**：ESP-IDF v5.5.2 · esp-tflite-micro 1.3.4 · esp-nn 1.2.5 · ESP32-C5 · Sensairshuttle `gesture_recognition`
> **记录时间**：2026-11
> 内容以实测和源码为准，版本更新后部分 API 可能变化；配套 esp-dl 笔记见 [esp32-p4 分支](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-p4)。

---

## 目录

- [1. esp-nn 是什么](#1-esp-nn-是什么)
- [2. 模型与部署基础（初学者）](#2-模型与部署基础初学者)
- [3. 从数据到芯片的完整流程](#3-从数据到芯片的完整流程)
- [4. 实战：ESP32-C5 手势识别项目](#4-实战esp32-c5-手势识别项目)
- [5. TFLite Micro 运行时与 esp-nn](#5-tflite-micro-运行时与-esp-nn)
- [6. ESP32-C5 平台配置](#6-esp32-c5-平台配置)
- [7. 性能与量化（float32 vs int8）](#7-性能与量化float32-vs-int8)
- [8. 实战踩坑（重点）](#8-实战踩坑重点)
- [9. 自定义手势与模型更新](#9-自定义手势与模型更新)
- [10. 学习路线与速查命令](#10-学习路线与速查命令)
- [11. 参考](#11-参考)

---

## 1. esp-nn 是什么

**esp-nn 是 Espressif 的 int8 神经网络算子库**，不是推理框架，也不是模型格式。

它做的事很单一：把卷积、深度卷积、池化、全连接、softmax 等算子，用 C / 汇编 / SIMD 写成芯片上的高性能实现。模型怎么解析、张量怎么分配、算子怎么调度，都不归它管——那是上层运行时（主要是 TensorFlow Lite Micro）的事。

几个关键事实：

- **它主要服务于 TFLite Micro**。官方 README 的 Supported platforms 只写了 `TensorFlow Lite Micro (TFLite Micro)`。
- **内核以 int8（s8）为主**，部分 s16；不是通用 float 推理库。
- **按芯片分级优化**：ESP32-S3 / ESP32-P4 有汇编/SIMD 专用路径；ESP32、ESP32-C3、ESP32-C5 等走通用优化或 ANSI C 实现。
- **不依赖 esp-dl**。esp-dl 自带算子实现，和 esp-nn 是两条并列的路线（见下表）。

### esp-nn 在 TinyML / TFLite 生态中的位置

**TinyML = Tiny Machine Learning（微型机器学习）**，指在 MCU 和超低功耗设备上直接跑推理。它是一个**领域**，不是具体框架。ESP32 上有两条常见路线：

```
TinyML（领域）
├── 路线 A：TFLite Micro + esp-nn    ← 本项目走的路线
└── 路线 B：esp-dl + esp-ppq         ← P4 笔记里的路线
```

| 名称 | 是什么 | 不是什么 |
|---|---|---|
| **TinyML** | 领域：资源受限 MCU 上的本地推理 | 不是某个具体框架 |
| **TFLite Micro (TFLM)** | 运行时 + 模型格式（`.tflite`，FlatBuffers） | 不是算子库，也不是 TinyML 的同义词 |
| **esp-nn** | Espressif 的 int8 算子库，TFLM 在 ESP32 上的加速后端 | 不是运行时，不解析模型 |
| **esp-dl** | Espressif 的另一套运行时 + `.espdl` 格式 + 模型库 | 不依赖 esp-nn，走自己的 SIMD/ISA |

### 两条路线对比

| 维度 | TFLite Micro + esp-nn | esp-dl + esp-ppq |
|---|---|---|
| 模型格式 | `.tflite`（FlatBuffers） | `.espdl`（FlatBuffers） |
| 量化工具 | TFLite Converter | esp-ppq（KL/percentile、TQT 等） |
| 运行时 | TFLM 解释器 | esp-dl execution plan |
| 算子 | esp-nn kernel + TFLM 参考实现 | esp-dl Module + 芯片 SIMD |
| 模型来源 | TFLite 生态（大量现成模型） | ESP-DL Model Zoo |
| 芯片支持 | ESP32 系列（优化程度不同） | S3 / S31 / P4 等 |
| 优势 | 生态大、跨平台、上手快 | 量化工具链深、检测后处理现成、P4 优化好 |
| 劣势 | 优化依赖 esp-nn 的芯片覆盖 | 生态窄、自定义算子成本高 |

> 一句话：**TFLM 是“运行时 + 格式”，esp-nn 是它下面的“算子实现库”；esp-dl 是另一套完整方案。**

---

## 2. 模型与部署基础（初学者）

> 这一节先讲“一个模型到底是什么”，再讲它怎么被塞进 MCU。有基础的读者可以直接跳到第 3 节。

### 2.1 一个模型 = 结构 + 参数

训练出来的“模型”可以拆成两部分：

1. **结构（计算图 / graph）**：由**算子（operator / layer）**和连接组成。常见算子：`Conv`、`FullyConnected`、`ReLU`、`MaxPool`、`Softmax`、`Reshape`……
2. **参数（parameters / weights）**：算子内部可学习的数值，主要是**权重 weight** 和**偏置 bias**。

另外几个必须区分的概念：

- **激活值（activation）**：推理时每层算出来的中间结果，不是训练出来的参数，随输入变化。
- **中间张量（intermediate tensor）**：除输入/输出外的所有张量，是运行时内存占用的主要来源。
- **超参数（hyperparameter）**：人定的，比如学习率、窗口长度、类别数。

### 2.2 推理 vs 训练

| | 训练（training） | 推理（inference） |
|---|---|---|
| 计算方向 | 前向 + 反向传播 | 只有前向 |
| 需要什么 | 梯度、损失函数、优化器、大量数据 | 一次输入 → 一组输出 |
| 产物 | 权重（`.pth` / `.h5`） | 可部署模型（`.tflite` / `.espdl`） |

TFLite Micro 只做推理，不做训练。

> **💡 打个比方：训练像射箭**
>
> 1. **前向传播（射箭）**：靠当前姿势和肌肉记忆（权重）把箭射出去，落在靶子左下角（预测）。
> 2. **计算误差（Loss）**：教练说偏了 10 厘米。
> 3. **反向传播（复盘）**：从结果倒推各部位的责任大小（梯度）。
> 4. **权重更新**：修正姿势，准备下一次。
>
> **推理只做第 1 步**：箭射出去就结束，不算误差、不更新权重。

### 2.3 张量与内存布局

**张量 = 多维数组 + 数据类型 + 形状 + 内存布局。**

- 本项目输入形状是 `[200, 3]`：**200 个时间步，每步 3 个轴**（gx, gy, gz）。
- 训练时 reshape 成 `(200, 3)`，端侧 `memcpy` 也必须按同样顺序；写成 `(3, 200)` 会完全错乱。
- 数据类型：训练用 `float32`；量化后用 `int8`（权重/激活）+ `int32`（bias 累加）。
- TFLite 内部可能把 1D 算子转成 2D 布局（见 4.5），但**你喂给输入张量的逻辑顺序不变**。

### 2.4 常见模型文件格式

| 格式 | 里面有什么 | 能直接部署到 TFLM 吗 |
|---|---|---|
| PyTorch `.pth` | 只有权重，结构在 Python 代码里 | 不能 |
| ONNX `.onnx` | 结构 + 权重 + 输入输出签名 + opset，自包含 | 不能，需转 TFLite |
| TensorFlow `.h5` / SavedModel | 结构 + 权重 | 不能，需转 TFLite |
| **`.tflite`** | TFLite 图 + 权重 + 量化参数 | 能 |
| **`.espdl`** | esp-dl 图 + int8 权重 + 量化参数 | 只能给 esp-dl |

### 2.5 量化概念

量化就是把 float32 映射到 int8：

```
量化：    q = round(x / scale) + zero_point
反量化：  x ≈ (q - zero_point) * scale
```

- **scale** 决定 int8 能表示的数值范围（`scale × 127` 是上限）。
- **per-tensor**：整个张量共用一个 scale；**per-channel**：每个通道一个 scale，精度更好。
- **校准（calibration）**：用一批真实输入跑 float 模型，统计每层激活范围，算出 scale。这就是“代表性数据集”的作用。
- TFLite 全整数量化需要 `representative_dataset`，否则只能做动态范围量化或保持 float。

### 2.6 FlatBuffers：文件格式也是内存布局

`.tflite` 和 `.espdl` 都基于 **FlatBuffers**。它在序列化时就按内存对齐和偏移量排好数据：

- 文件里的样子 ≈ 内存里的样子；
- 运行时不需要“解析成对象树”，把基地址交给程序，按偏移读字段即可；
- 对 MCU 意味着省 RAM、省加载时间。

> 类比：
> 普通格式（Protobuf/JSON）= 乐高装在盒子里，玩之前要倒出来按图纸拼；
> FlatBuffers = 出厂已拼好的乐高，打开盒子直接玩。

**但“允许零拷贝”不等于“一定零拷贝”**：本项目把模型作为 `const` 数组编进 Flash，TFLM 直接按偏移访问；如果模型放外部存储/PSRAM，是否拷贝取决于加载方式。

### 2.7 一次推理在 TFLM 里的生命周期

```
1. 加载    tflite::GetModel(model_tflite) 拿到 FlatBuffer 指针
2. 注册    MicroMutableOpResolver<8> 注册模型用到的算子
3. 分配    MicroInterpreter + tensor_arena，AllocateTensors()
4. 填输入  memcpy(input->data.f, 200×3 float, 2400 字节)
5. 前向    interpreter->Invoke()
6. 取输出  output->data.f → argmax → 类别 + 置信度
```

没有动态内存分配，所有中间张量都在 `tensor_arena` 里复用。

### 2.8 给初学者的几个“常识校正”

- **“模型多大”和“运行占多少内存”是两回事**：权重放 Flash；运行时还要 arena 放激活/中间张量。本项目模型 11KB，arena 50KB。
- **推理速度不等于参数量**：卷积的计算量 = 输出尺寸 × 卷积核 × 输入通道，和输入长度强相关。
- **float32 模型能跑，但不代表高效**：当前 ESP32-C5 工程按软浮点构建，float 由 rvfplib 模拟（见第 6 节）。
- **模型不是“越准越好”**：端侧还要看 RAM、Flash、延迟、功耗。
- **模型文件不能跨芯片/跨运行时通用**：`.tflite` 给 TFLM，`.espdl` 给 esp-dl；量化目标不同也不能混用。

---

## 3. 从数据到芯片的完整流程

本项目的全链路：

```
手腕动作
  → BMI270 陀螺仪（200Hz，±2000dps）
  → ESP32-C5 采集 200 帧 × 3 轴（每 5ms 一帧，共 1 秒）
  → 串口输出 CSV（一行 600 个数）
  → PC：pandas 读入 → reshape (N, 200, 3) → 标签 0/1/2
  → Keras 训练 CNN（约 1427 参数）
  → TFLite Converter 转成 .tflite（11,072 字节，float32）
  → xxd -i 生成 model.cpp C 数组
  → 编进固件（Flash rodata）
  → TFLM：GetModel → MicroInterpreter → Invoke
  → argmax + 去抖 → 串口打印类别
```

关键约束：

- **输入形状固定**：`200 × 3`。改窗口长度必须重训模型。
- **batch = 1**：TFLM 只做单样本推理。
- **模型是 float32**：当前没有量化；想用 esp-nn 加速需要 int8。
- **推理完全离线**：不联网、不调用云端。

---

## 4. 实战：ESP32-C5 手势识别项目

### 4.1 项目结构

| 文件 | 作用 |
|---|---|
| `main/motion_estimation.cpp` | `app_main`：初始化按键、IMU、模型，启动流水线 |
| `main/app_btn.cpp` | BOOT 按键（GPIO28）：短按触发采集，长按 2s 切换模式 |
| `main/app_bmi270.cpp` | I2C + BMI270 初始化，单次采样与 LSB→dps 转换 |
| `main/app_model.cpp` | TFLM 解释器、算子注册、arena、推理与 argmax |
| `main/app_model_pipeline.cpp` | collect / inference 两个 FreeRTOS 任务，滑窗与去抖 |
| `main/model.cpp` | `xxd -i` 生成的模型 C 数组（11,072 字节） |
| `main/Kconfig.projbuild` | 引脚、采样率、窗口、阈值、滑窗步长等参数 |
| `main/idf_component.yml` | esp-tflite-micro / bmi270_sensor / button 依赖 |

### 4.2 硬件与引脚

| 信号 | GPIO | 说明 |
|---|---|---|
| I2C SCL | 3 | 时钟线 |
| I2C SDA | 2 | 数据线 |
| SDO | 9 | I2C 地址选择，拉低 → `0x68` |
| BOOT 按键 | 28 | 模式切换/触发采集 |

- I2C 速率：200kHz。
- BMI270：陀螺仪 200Hz、±2000dps；加速度计 200Hz、±2g（**当前推理不用加速度**）。
- 数据就绪中断映射到 INT1，但代码实际是**轮询**状态寄存器。
- 开发板：Sensairshuttle（ESP32-C5），8MB Flash、8MB PSRAM。

### 4.3 数据采集：把动作变成 CSV

两种模式：

- **INFERENCE（默认）**：上电即进入，持续采样 + 滑窗推理。
- **COLLECT**：长按 BOOT 2s 切换；短按后等待角速度触发。

触发条件（仅 COLLECT 模式）：

```c
sumabs = |gx| + |gy| + |gz|;
if (sumabs > CONFIG_TRIGGER_THRESHOLD) {  // 默认 100 dps
    采集 200 帧;
}
```

采完通过 `printf` 输出一行 CSV：

```
12.50000,-3.20000,88.10000,13.00000,-2.90000,87.60000,...,10.10000,0.50000,90.00000
```

一行 600 个数 = 200 帧 × 3 轴。同一类动作保存到 `o.txt` / `v.txt` / `unknown.txt`，再转成 CSV。

**注意**：

- 日志行（`I (...) ...`）会混进串口输出，训练前必须过滤，只保留纯 CSV 行。
- `Unknown` 类很重要：采集各种“不是目标手势”的动作，否则模型会在圈/V 之间硬选。
- 换人、换手持方向、换速度各采一些，泛化性主要靠数据多样性。

### 4.4 训练：一个小 CNN

README 的数据处理：

```python
o_data       = pd.read_csv('./o.csv', header=None)
v_data       = pd.read_csv('./v.csv', header=None)
unknown_data = pd.read_csv('./unknown.csv', header=None)

X = np.vstack([o_data.values, v_data.values, unknown_data.values])
X = X.reshape(-1, 200, 3)          # 每行 600 个数 → 200×3
y = np.concatenate([
    np.zeros(len(o_data), dtype=int),
    np.ones(len(v_data), dtype=int),
    np.full(len(unknown_data), 2, dtype=int),
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

网络结构：

```python
model = keras.Sequential([
    keras.layers.Conv1D(8, 5, padding='same', activation='relu', input_shape=(200, 3)),
    keras.layers.MaxPooling1D(4),
    keras.layers.Conv1D(16, 5, padding='same', activation='relu'),
    keras.layers.MaxPooling1D(4),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(3, activation='softmax'),
])
```

| 层 | 输出形状 | 参数量 | 作用 |
|---|---:|---:|---|
| 输入 | 200×3 | 0 | 1 秒、3 轴角速度 |
| Conv1D(8, k=5) | 200×8 | 128 | 找局部波形 |
| MaxPool(4) | 50×8 | 0 | 降采样、容忍速度差异 |
| Conv1D(16, k=5) | 50×16 | 656 | 组合动作片段 |
| MaxPool(4) | 12×16 | 0 | 再降采样 |
| GlobalAvgPool | 16 | 0 | 压成特征向量 |
| Dense(32) | 32 | 544 | 特征组合 |
| Dropout(0.2) | 32 | 0 | 防过拟合 |
| Dense(3, softmax) | 3 | 99 | 三类概率 |

总计约 **1427 个参数**，float32 权重约 5.7KB。

训练配置：Adam(1e-3)、`sparse_categorical_crossentropy`、300 epochs、batch 64、shuffle、70/30 划分。

**评估建议**：

- 不要只看总准确率，画混淆矩阵；
- 重点看 Unknown 被误判成手势的比例（实际误报）；
- 随机划分可能把同一次采集的相邻窗口分到训练/测试，导致准确率虚高；更严谨的是按场次/人划分。

### 4.5 转换与部署

转换：

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_model = converter.convert()
```

得到 **11,072 字节**的 float32 `.tflite`。

**1D 算子会被降级成 2D**：

- `Conv1D` → `Conv2D` + `Reshape/ExpandDims`
- `MaxPooling1D` → `MaxPool2D`
- `GlobalAveragePooling1D` → `Mean`（或 `AveragePool2D` + `Reshape`）

所以固件注册的是：

```c
resolver.AddSoftmax();
resolver.AddFullyConnected();
resolver.AddMaxPool2D();
resolver.AddConv2D();
resolver.AddAveragePool2D();
resolver.AddExpandDims();
resolver.AddReshape();
resolver.AddMean();
```

一共 8 个，对应 `MicroMutableOpResolver<8>`。

**xxd 转换**：

```bash
xxd -i model.tflite > model.cpp
```

仓库里的 `model.cpp` 还加了：

```c
alignas(8) const unsigned char model_tflite[] = { ... };
const unsigned int model_tflite_len = 11072;
```

三个细节：

1. **保留 `const`**：否则数组会从 Flash 复制到 RAM，白占 11KB。
2. **保留 `alignas(8)`**：FlatBuffer 有对齐要求。
3. **符号名必须是 `model_tflite`**：`xxd -i` 的命名取决于文件名/路径，名字不对会链接失败。

**TFLM 初始化**：

```c
const tflite::Model *model = tflite::GetModel(model_tflite);
if (model->version() != TFLITE_SCHEMA_VERSION) return;

static tflite::MicroMutableOpResolver<8> resolver;
// ... AddXxx()

static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
interpreter->AllocateTensors();

input  = interpreter->input(0);
output = interpreter->output(0);
```

`tensor_arena` 是 50KB 静态数组，所有中间激活都放这里。

> **💡 打个比方**：模型像一本 11KB 的菜谱，存在 Flash；arena 是 50KB 的案板，切好的菜、半成品都摊在上面。菜谱只有几页，但案板要够大。

### 4.6 端侧流水线：两个任务 + 滑窗 + 去抖

`app_main` 启动后创建两个 FreeRTOS 任务：

| 任务 | 优先级 | 栈 | 职责 |
|---|---:|---:|---|
| `collect` | 4 | 5KB | 每 5ms 采样，填满 200×3 窗口 |
| `inference` | 5 | 6KB | 复制窗口，跑模型，去抖，打印 |

交接方式：

```
collect: 填满 200 帧 → xQueueSend(指针) → 等信号量
inference: xQueueReceive → memcpy 到自己的 buffer → 给信号量 → Invoke
```

**关键设计**：推理任务先 `memcpy` 再给信号量，所以采样任务只在复制那一瞬间等待，不会被整个推理过程阻塞。

**滑窗**：

```
[帧 0 .. 199]  → 推理
[帧 30 .. 229] → 推理   （保留 170 帧，补 30 帧）
[帧 60 .. 259] → 推理
```

`CONFIG_SLIDING_WINDOW_STEP = 30`，30 × 5ms = **150ms** 一次推理。

**去抖状态机**：

- 置信度 < 85%：忽略；
- 同类连续 3 次：打印一次；
- 同类连续 5 次：重置，准备识别下一个动作；
- 类别变化：重新计数。

> **💡 打个比方**：三个评委都亮同一盏灯才宣判；播报一次后，要等这阵动作过去再听下一句。

### 4.7 一次识别的时间线

```
t=0ms      静止：gx=1.2, gy=-0.8, gz=2.1，和=4.1（COLLECT 不触发）
t=50ms     开始画圈：gx=150, gy=-40, gz=90，和=280
           → COLLECT：触发，开始录 200 帧
           → INFERENCE：持续采样，无需触发
t=1050ms   200 帧录满 → 模型输出 [0.97, 0.01, 0.02] → 圈 97%
           连续 3 次高置信度 → 打印
t=1200ms   滑窗 30 帧，继续推理
```

串口输出示例：

```
I (9992) Model Pipeline: Motion detected: Counterclockwise circle (confidence: 100.0%)
I (13232) Model Pipeline: Motion detected: V (confidence: 99.9%)
```

---

## 5. TFLite Micro 运行时与 esp-nn

### 5.1 TFLM 的四个核心组件

| 组件 | 作用 |
|---|---|
| `tflite::Model` | FlatBuffer 模型指针，零拷贝访问 |
| `MicroMutableOpResolver<N>` | 注册模型用到的算子，N 是容量 |
| `MicroInterpreter` | 调度算子，管理张量生命周期 |
| `tensor_arena` | 一块固定 RAM，所有中间张量都在这 |

TFLM 不做训练，不做动态内存分配（arena 用完后不再申请）。

### 5.2 为什么只注册 8 个算子？

TFLM 是“按需编译”的思路：只把模型用到的算子链接进来，减小固件体积。如果模型用了新算子而 resolver 没注册，`AllocateTensors()` 或 `Invoke()` 会报：

```
Didn't find op for builtin opcode 'CONV_2D' version '3'
```

### 5.3 float32 和 int8 走的是不同内核

这是本项目最关键的技术点之一。

看 `esp-tflite-micro` 的卷积实现
`tensorflow/lite/micro/kernels/esp_nn/conv.cc`：

```c
// int8 分支
if (input->type == kTfLiteInt8) {
    esp_nn_conv_s8(...);                 // ← esp-nn 优化内核
}

// float32 分支
case kTfLiteFloat32: {
    tflite::reference_ops::Conv(...);    // ← TFLM 自带通用实现
}
```

- **int8 模型** → 调用 `esp_nn_conv_s8`；
- **float32 模型** → 走 `reference_ops::Conv`，**不会调用 esp-nn**。

本项目模型是 float32（`input->data.f`），所以：

> **esp-nn 虽然被编译进固件了，但它的优化内核没有在这个模型的热路径上执行。**

“没用”不是“坏了”，是**接口对不上**。

### 5.4 为什么固件里还有 esp-nn？

因为 `esp-tflite-micro` 在组件清单里依赖 `esp-nn`；构建时它会把原来的 `conv.cc`、`pooling.cc`、`fully_connected.cc`、`softmax.cc` 等替换成带 esp-nn 分支的版本。这个版本同时包含 int8 和 float 分支，所以库被链接进来，但 float 模型走的是 float 分支。

### 5.5 怎么确认有没有真的调用？

- **看代码分支**：最直接；
- **看符号**：`nm build/motion_estimation.elf | grep esp_nn` 只能证明“链接了”，不能证明“执行了”；
- **看耗时**：在 `Invoke()` 前后用 `esp_timer_get_time()` 打点，对比 float 和 int8 两个模型；
- **看 profiling**：用 `esp_timer` 打点，或接 TFLM 的 `MicroProfiler`（在解释器构造时传入）。

### 5.6 ESP32-C5 上的 esp-nn 优化程度

esp-nn 的汇编/SIMD 专用路径主要覆盖 **ESP32-S3** 和 **ESP32-P4**。ESP32-C5 不在专用汇编列表里，走的是**通用优化 C** 或 ANSI C 实现。

所以：

- int8 在 C5 上仍然比 float 软浮点快、体积小；
- 但不要期待 S3/P4 那种“十几倍”的汇编加速；
- 具体收益以实测为准。

---

## 6. ESP32-C5 平台配置

### 6.1 芯片与工程参数

| 项 | 值 |
|---|---|
| CPU | RISC-V 单核，240MHz |
| Flash | 8MB（`sdkconfig.defaults` 配置） |
| PSRAM | 8MB，80MHz |
| FreeRTOS tick | 1000Hz |
| 工程名 | `motion_estimation` |
| 目标 | `esp32c5` |

启动日志（节选）：

```
cpu_start: Unicore app
cpu_start: cpu freq: 240000000 Hz
esp_psram: Found 8MB PSRAM device
app_init: Project name: motion_estimation
```

### 6.2 软浮点：float 由 rvfplib 模拟

从构建参数可以看到：

```
-march=rv32imac_zicsr_zifencei
```

`rv32imac` 里**没有 `f`**（F = 单精度浮点扩展），说明编译器不生成硬件浮点指令。`sdkconfig` 里：

```
CONFIG_COMPILER_FLOAT_LIB_FROM_RVFPLIB=y
```

即使用乐鑫的 `rvfplib` 软件浮点库。它比 GCC 通用软浮点快，但本质上仍是整数指令模拟。

**影响**：

- float32 推理**能跑**，但比 int8 慢、占代码空间；
- 小模型（本项目）没问题；
- 大模型（MobileNet 级别）会非常吃力，int8 几乎是必选项。

> **💡 打个比方**：没有计算器，但带了一本熟练的算盘手册。结果一样，速度慢一些。

### 6.3 arena 放哪里？

`tensor_arena` 是普通静态数组，默认放**内部 RAM**，不是 PSRAM。8MB PSRAM 虽然大，但：

- PSRAM 访问延迟高于内部 RAM；
- 把 arena 放 PSRAM 需要额外配置/链接脚本；
- 本项目 50KB arena 放内部 RAM 完全够。

可用 `interpreter->arena_used_bytes()` 看实际用量，再决定是否调整 `kTensorArenaSize`。

### 6.4 构建与烧录

```bash
. $IDF_PATH/export.sh
idf.py set-target esp32c5
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

菜单配置：

```bash
idf.py menuconfig
# → Motion Estimation
```

---

## 7. 性能与量化（float32 vs int8）

### 7.1 模型规模

| 项 | 数值 |
|---|---:|
| 参数量 | 约 1427 |
| 乘加次数 | 约 5.7 万次（粗算） |
| 模型文件 | 11,072 字节 |
| 输入 | 200×3 float = 2.4KB |
| arena | 50KB |
| 推理频率 | 每 150ms 一次 |

按这个规模，即使软浮点，单次推理通常也是**几毫秒到十几毫秒**量级；具体以 `esp_timer` 实测为准。相对 150ms 的窗口步长，余量很大。

### 7.2 float32 vs int8

| | float32 | int8 |
|---|---|---|
| 权重/激活 | 4 字节/个 | 1 字节/个 |
| 运算 | 软浮点（当前工程无 FPU 指令） | 整数乘加 |
| esp-nn | 不走优化内核 | 走 esp-nn s8 内核 |
| 速度 | 较慢 | 更快 |
| 精度 | 通常略好 | 需校准，可能略降 |
| 部署难度 | 简单 | 需代表性数据集 + 改输入输出类型 |
| 适合 | 小模型、低频率、原型 | 较大模型、高帧率、省电 |

### 7.3 什么时候该量化？

判断标准：

- `Invoke()` 耗时接近或超过窗口步长（150ms）；
- CPU 占用过高，影响采样节拍；
- 模型变大，Flash/RAM 吃紧；
- 想省电。

测量方法：

```c
int64_t t0 = esp_timer_get_time();
interpreter->Invoke();
int64_t t1 = esp_timer_get_time();
ESP_LOGI(TAG, "invoke = %lld us", t1 - t0);
```

### 7.4 int8 转换要点

TFLite 全整数量化大致是：

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
```

端侧要跟着改：

- 输入：`input->data.int8`，按模型 scale/zero_point 量化；
- 输出：`output->data.int8`，反量化回概率；
- `memcpy` 长度和数据类型都要改；
- 确认 `esp_nn_conv_s8` 等符号真的被调用。

> ⚠️ 量化不是“无脑压缩”：校准集要覆盖真实输入分布；量化后要用混淆矩阵验证，尤其是 Unknown 类。

---

## 8. 实战踩坑（重点）

### 8.1 改了 `NUM_SAMPLES` 却没重训模型

**症状**：输入数据错位、推理结果乱飞，甚至内存越界。

**原因**：模型输入张量固定 `200×3`，但端侧 `memcpy` 按 `num_samples * 3 * sizeof(float)` 拷贝。`NUM_SAMPLES` 改成 300 就会写越界；改成 100 则只填了一部分。

**解决**：窗口长度、采样间隔、模型输入三者必须一起改、一起重训。

### 8.2 标签顺序或数量不一致

**症状**：模型明明认出圈，打印成 V；或者数组越界。

**原因**：训练标签 `0=圈, 1=V, 2=Unknown`，固件 `kLabels[]` 必须同序同长度。`app_model_predict()` 的 `num_classes` 是从输出张量读的，但 `kLabels` 是硬编码的 3 项。

**解决**：加类别时同步改训练标签、`Dense(N+1)`、`kLabels`。

### 8.3 算子没注册

**症状**：

```
Didn't find op for builtin opcode '...' version '...'
```

**原因**：换模型后用了新的 TFLite 算子，但 `MicroMutableOpResolver<8>` 没注册。

**解决**：在 `app_model_init()` 里 `AddXxx()`，必要时把模板容量从 8 调大。

### 8.4 arena 太小

**症状**：`AllocateTensors() failed`。

**原因**：中间张量总需求超过 `kTensorArenaSize`。

**解决**：用 `interpreter->arena_used_bytes()` 看实际用量，调大 50KB；不要盲目设几百 KB，内部 RAM 有限。

### 8.5 输入布局/类型不匹配

**症状**：模型能跑但准确率极低。

**原因**：

- 训练 reshape 成 `(200, 3)`，端侧按 `(3, 200)` 填；
- 模型是 int8，端侧却用 `input->data.f`；
- 模型是 float32，端侧却按 int8 访问。

**解决**：训练/端侧共用同一套布局和 dtype 约定；量化模型要同步改代码。

### 8.6 采样节拍与 ODR 抖动

**症状**：同样的动作，有时识别、有时不识别；数据分布和训练集对不上。

**原因**：

- 代码用 `vTaskDelay(5ms)` 软件延时，不是硬件定时器；
- BMI270 ODR 200Hz，轮询可能偶尔没准备好，代码会重试；
- 实际采样间隔有抖动。

**解决**：训练数据和推理数据用同一套采集方式；如果对时间精度要求高，改用 DRDY 中断/硬件定时器打时间戳。

### 8.7 CSV 日志污染

**症状**：训练脚本读 CSV 报错，或数据形状不对。

**原因**：串口同时输出 `ESP_LOGI` 日志行和纯 CSV 行。

**解决**：采集时过滤掉 `I (...)` / `W (...)` / `E (...)` 行，只保留数字和逗号的行。

### 8.8 数据划分泄漏

**症状**：测试准确率很高，实机却经常错。

**原因**：随机 70/30 划分把同一次采集的相邻滑窗分到了训练集和测试集，二者高度相似。

**解决**：按采集场次/人/时间划分；保留一个独立的实机测试集。

### 8.9 推理模式其实没有触发阈值

**症状**：以为“静止不推理”，实际一直在跑。

**原因**：README 说推理模式在角速度和超过阈值时才推理，但当前代码的 INFERENCE 分支是**持续采样 + 持续滑窗推理**，没有阈值判断；阈值只在 COLLECT 模式生效。

**解决**：如果在意功耗，自己加触发门控；如果只是识别，持续推理响应更连续。

### 8.10 按键信号量残留

**症状**：切回 COLLECT 模式后，还没按键就进入了“等待触发”状态。

**原因**：INFERENCE 模式下短按也会释放二值信号量；切回 COLLECT 时可能立刻消费这次点击。

**解决**：切模式时清空/重置信号量，或只在 COLLECT 模式注册短按回调。

### 8.11 初始化失败不传播

**症状**：程序正常启动，但推理一直失败。

**原因**：`app_model_init()` 是 `void`，模型 schema 不匹配或 `AllocateTensors()` 失败只打印日志，流水线仍会启动。

**解决**：把初始化改成返回 `esp_err_t`，失败时停止创建任务；先看启动日志。

### 8.12 `xxd` 生成的文件丢了 `const` / `alignas`

**症状**：模型占用 RAM、对齐异常，甚至跑飞。

**原因**：`xxd -i` 默认生成非 `const` 的数组；直接覆盖会丢掉仓库里手工加的 `alignas(8)` 和 `const`。

**解决**：替换时保留：

```c
alignas(8) const unsigned char model_tflite[] = { ... };
const unsigned int model_tflite_len = ...;
```

并确认符号名是 `model_tflite`。

### 8.13 误以为 float32 模型在用 esp-nn 加速

**症状**：看到依赖里有 esp-nn，就以为推理走了优化内核。

**真相**：float32 卷积走 `reference_ops::Conv`，esp-nn 的 `esp_nn_conv_s8` 只在 int8 分支被调用。想用 esp-nn，先量化到 int8。

### 8.14 模型外置时的对齐与生命周期

**症状**：从分区/SD 加载模型后，`GetModel()` 崩溃或推理结果异常。

**原因**：

- 读入的 buffer 没满足 FlatBuffer 对齐要求；
- buffer 在推理前被释放/复用；
- 读取长度和 `model_tflite_len` 不一致。

**解决**：用静态或 `aligned_alloc` 的缓冲区，保持到推理结束；校验文件长度和 schema 版本。

---

## 9. 自定义手势与模型更新

### 9.1 当前模型是“固定类别”

当前固件只能识别训练过的 3 类：圈、V、Unknown。训练完权重就冻结了，推理只做前向，不会自学新手势。

- 新手势不会被自动识别；
- 大概率被判成 Unknown，也可能误判成圈/V；
- `Unknown` 不是万能识别器，只是训练出来的兜底类。

### 9.2 自定义手势 = 重新训练 + 重新部署

流程：

| 步骤 | 做什么 |
|---|---|
| 1 | 定义 N 个手势 + 1 个 Unknown |
| 2 | 每个手势采集几百条，换人/方向/速度 |
| 3 | 最后一层改 `Dense(N+1, softmax)`，重排标签 |
| 4 | 重训、转 `.tflite`、`xxd -i` 替换 `model.cpp` |
| 5 | 同步改 `kLabels[]`，保持顺序一致 |
| 6 | 必要时加大 arena、注册新算子 |
| 7 | `idf.py build flash` |

### 9.3 三种“更新模型”的方式

| 方式 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| 整包重烧 | 替换 `model.cpp`，重编固件 | 最简单 | 需要重新烧整个固件 |
| 模型外置 | `.tflite` 放 Flash 分区 / SD 卡，运行时加载 | 只换模型，可 OTA/版本管理 | 要改加载逻辑、对齐、生命周期 |
| 端侧学习 | MCU 上训练/微调 | 用户现场教新手势 | TFLM 不支持；成本和难度高 |

**模型外置的伪代码**：

```c
uint8_t *model_buf = read_model_from_partition();  // 静态/aligned 缓冲
const tflite::Model *model = tflite::GetModel(model_buf);
// model_buf 必须活到推理结束
```

模型只有 11KB，放一个 64KB 分区绰绰有余。

### 9.4 不重训能不能扩展？

严格来说不能真正增加类别。可行的替代：

- 用现有模型倒数第二层做特征提取，PC 上训练新分类头（仍需部署新模型）；
- 模板匹配 / DTW：录模板即可加手势，不用训练，但鲁棒性和内存开销不同；
- 应用层组合规则：比如“连续两次 V”触发动作，这不是新手势。

---

## 10. 学习路线与速查命令

### 10.1 建议学习路线

| 阶段 | 目标 | 关键内容 |
|---|---|---|
| 0 | 搞清概念 | TinyML、TFLM、esp-nn、esp-dl 的层级关系 |
| 1 | 跑通示例 | `idf.py build flash monitor`，看推理日志 |
| 2 | 理解数据 | 采集模式、CSV、触发阈值、窗口/滑窗 |
| 3 | 理解模型 | CNN 结构、参数量、输入输出、argmax |
| 4 | 走通训练/部署 | pandas reshape → Keras → TFLite → xxd |
| 5 | 改手势 | 采新数据、改 `Dense(N+1)`、改 `kLabels` |
| 6 | 优化 | float vs int8、esp-nn 内核、arena、功耗 |
| 7 | 工程化 | 模型分区/OTA、版本管理、异常处理 |

### 10.2 速查命令

```bash
# 环境与构建
. $IDF_PATH/export.sh
idf.py set-target esp32c5
idf.py build
idf.py -p /dev/ttyACM0 flash monitor

# 菜单配置
idf.py menuconfig   # → Motion Estimation

# 查看模型/固件大小
idf.py size
idf.py size-components

# 转换模型
xxd -i model.tflite > model.cpp

# 查看 esp-nn 符号（只能证明链接，不能证明执行）
nm build/motion_estimation.elf | grep esp_nn

# 反汇编/地址解析
riscv32-esp-elf-addr2line -pfiaC -e build/motion_estimation.elf 0x...
```

### 10.3 关键数字

| 项 | 值 |
|---|---:|
| 采样间隔 | 5ms |
| 窗口 | 200 帧 × 3 轴 = 1 秒 |
| 滑窗步长 | 30 帧 = 150ms |
| 触发阈值 | \|gx\|+\|gy\|+\|gz\| > 100 dps |
| 置信度门槛 | 85% |
| 去抖 | 连续 3 次确认，5 次后重置 |
| 模型 | 约 1427 参数，11,072 字节 |
| arena | 50KB |
| 类别 | 圈 / V / Unknown |

---

## 11. 参考

- esp-tflite-micro：<https://github.com/espressif/esp-tflite-micro>
- esp-nn：<https://github.com/espressif/esp-nn>
- TFLite Micro 文档：<https://ai.google.dev/edge/litert/microcontrollers/overview>
- ESP-IDF 编程指南：<https://docs.espressif.com/projects/esp-idf/>
- 本项目示例：`esp-sensairshuttle/examples/gesture_recognition`
- 配套 esp-dl 笔记：[esp32-p4-notes](https://github.com/yh-xh/esp32-p4-notes)

---

> 本文记录来自 ESP32-C5 手势识别项目的实际调试，欢迎提 Issue 交流或补充。
