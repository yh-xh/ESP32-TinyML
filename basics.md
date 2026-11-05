# 模型与部署基础（公共篇）

> 这是 [ESP32-P4 / esp-dl 笔记](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-p4) 和
> [ESP32-C5 / esp-nn 笔记](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-c5) 的**公共基础**：
> 模型是什么、怎么量化、怎么装进 MCU。
> 两份笔记的第 2 章都指向这里，平台特有的内容各自保留。
>
> 想先补神经网络本身（神经元、激活函数、CNN、TCN），看
> [neural-networks.md](https://github.com/yh-xh/ESP32-TinyML/blob/main/neural-networks.md)。

---

## 目录

- [1. 一个模型 = 结构 + 参数](#1-一个模型--结构--参数)
- [2. 推理 vs 训练](#2-推理-vs-训练)
- [3. 张量与内存布局](#3-张量与内存布局)
- [4. 常见模型文件格式](#4-常见模型文件格式)
- [5. 量化概念](#5-量化概念)
  - [5.1 PTQ 与 TQT：量化精度不够时怎么办](#51-ptq-与-tqt量化精度不够时怎么办)
- [6. FlatBuffers：文件格式也是内存布局](#6-flatbuffers文件格式也是内存布局)
- [7. 给初学者的常识校正](#7-给初学者的常识校正)
- [8. 两条实现路线](#8-两条实现路线)

---

## 1. 一个模型 = 结构 + 参数

训练出来的"模型"可以拆成两部分：

1. **结构（计算图 / graph）**：由**算子（operator / layer）**和它们之间的连接组成。
   常见算子：`Conv`（卷积）、`Gemm`/`Linear`（全连接）、`ReLU`/`Sigmoid`（激活）、
   `MaxPool`（池化）、`Concat`（拼接）、`Reshape`、`Softmax`……
   结构决定"数据怎么流动"：输入 → 算子 1 → 算子 2 → … → 输出。
2. **参数（parameters / weights）**：算子内部可学习的数值，主要是**权重 weight** 和**偏置 bias**。

另外几个必须区分的概念：

- **激活值（activation）**：推理时每层算出来的**中间结果**，不是训练出来的参数，**随输入变化**。
  量化时"激活的 scale"要靠校准数据统计，原因就在这里。
- **中间张量（intermediate tensor）**：除输入/输出之外的所有张量，是运行时内存占用的主要来源。
- **超参数（hyperparameter）**：不是学出来的，是人定的，例如学习率、输入尺寸、窗口长度、类别数。
- **算子 vs 层**：日常混用；严格说"层"是结构概念，"算子"是计算概念
  （一个 Conv + Bias + ReLU 可能被融合成一个算子）。

## 2. 推理 vs 训练

| | 训练（training） | 推理（inference） |
|---|---|---|
| 计算方向 | 前向 + 反向传播 | 只有前向 |
| 需要什么 | 梯度、损失函数、优化器、大量数据 | 一次输入 → 一组输出 |
| 产物 | 权重（`.pth` / `.h5` 等） | 可部署模型（`.tflite` / `.espdl`） |

部署时只保留**前向图**，所以部署文件通常比训练检查点小得多。MCU 上的运行时
（TFLM / esp-dl）都只做推理，不做训练。

> **💡 打个比方：训练是怎么发生的（射箭）**
>
> 1. **前向传播（射箭）**：你（输入数据）靠当前的姿势和肌肉记忆（**权重**）把箭射出去，
>    箭落在靶子左下角（**输出预测**）。
> 2. **计算误差（Loss）**：教练告诉你，离靶心偏了 10 厘米。
> 3. **反向传播（复盘）**：教练从结果倒推——"手臂抬低了 2°、呼吸没稳住"，
>    算出身体各部位（**各权重**）对偏差应负的"责任大小"（**梯度**）。
> 4. **权重更新**：按建议修正角度和呼吸（**更新权重**），准备下一次。
>
> **推理就是只做第 1 步**：箭射出去就结束——不算误差、不更新权重。
> 这也解释了为什么部署的模型只需要"前向图"。

## 3. 张量与内存布局

**张量（tensor）** = 多维数组 + 数据类型 + 形状 + 内存布局。

- **形状**：图像模型常见 `[N, C, H, W]`（batch、通道、高、宽），时序模型常见 `[T, F]`
  （时间步、特征）。**端侧运行时基本都只支持 batch = 1**。
- **数据类型**：训练用 `float32`；量化后用 `int8`（权重/激活）+ `int32`（bias 累加）。
- **内存布局**：同一张图，`NCHW` 和 `NHWC` 在内存里的排法不同；做颜色转换、resize、
  `memcpy` 时必须和训练时一致。
- **步长 stride**：一行占多少字节。`stride = 宽 × 每像素字节数`（有些硬件会对齐到 16/64 字节）。
  访问第 y 行第 x 列 = `data + y×stride + x×bpp`；stride/布局不对，整幅数据都会错位。
- **像素格式**也是一种布局：摄像头常输出 `RGB565`（2 字节/像素），训练常用 `RGB888`（3 字节）。

两个实际例子：

- 图像（P4 项目）：输入是 `1024×600` 的 `RGB565` 帧，模型前处理会转换成模型要求的尺寸/格式。
- 时序（C5 项目）：手势输入是 `[200, 3]`（200 个时间步 × 3 轴），训练时怎么 reshape，
  端侧 `memcpy` 就必须按同样顺序；写成 `(3, 200)` 会完全错乱。

## 4. 常见模型文件格式

| 格式 | 里面有什么 | 说明 |
|---|---|---|
| PyTorch `.pth` | **只有权重**，结构在 Python 代码里 | 需要 PyTorch 解释器，不能直接部署 |
| ONNX `.onnx` | 结构 + 权重 + 输入输出签名 + opset，**自包含** | 通用交换格式，需再转目标运行时格式 |
| TensorFlow `.h5` / SavedModel | 结构 + 权重 | 需转 TFLite |
| TensorFlow Lite `.tflite` | 图 + 权重 + 量化参数（FlatBuffers） | 给 TFLite / TFLite Micro |
| `.espdl` | esp-dl 图 + int8 权重 + 量化参数（FlatBuffers） | 给 esp-dl |

> 关键点：`.pth` 和 `.onnx` 的区别不是"是不是权重文件"，而是 **ONNX 自包含**——
> `.onnx` 的 `graph.initializer` 里就存着训练好的权重。

## 5. 量化概念

量化就是把 float32 的数值映射到 int8：

```
量化：    q = round(x / scale) + zero_point
反量化：  x ≈ (q - zero_point) * scale
```

- **`scale`** 决定 int8 能表示的数值范围（`scale × 127` 是上限）。
- **per-tensor**：整个张量共用一个 scale；**per-channel**：每个通道一个 scale，精度更好。
- **校准（calibration）**：用一批真实输入跑一遍 float 模型，统计每层激活值的范围 →
  算出激活的 scale。数据集要有代表性，否则 scale 会定得离谱。
- **不同运行时的细节**：
  - esp-dl 用 **`scale = 2^exponent`**（2 的幂）表示，反量化可用移位/查表实现；
    `.info` 里看到的 `exponents: [-4]` 就是 `scale = 2^-4 = 0.0625`。
  - TFLite 全整数量化需要提供 `representative_dataset`，否则只能做动态范围量化或保持 float。

**量化误差从哪来**

| 来源 | 说明 | 缓解 |
|---|---|---|
| **舍入误差** | 每个数最多差 `scale/2`；scale 越小误差越小，但可表示范围也越小 | 用 per-channel、16 bit |
| **截断误差** | 超出 `scale×127` 的激活被截在边界上 | KL / percentile 校准，裁掉长尾 |
| **per-channel** | 权重按输出通道分别定 scale（`scale_c = max|w_c| / 127`） | 避免一个通道的极值拖累其他通道 |
| **累加精度** | int8×int8 用 **int32 累加**，防止溢出 | 框架自动处理；注意大模型累加和 |
| **2 的幂 scale** | esp-dl 用 `2^exponent` 省乘法，但 scale 只能取离散值 | 精度略低于任意 scale，换来更快的反量化 |

> 记住一个直觉：**量化掉点通常不是"平均误差"造成的，而是少数离群值把 scale 撑大，
> 导致正常数值只用到 int8 的一小部分**。这就是校准算法（KL/percentile）存在的意义。

### 5.1 PTQ 与 TQT：量化精度不够时怎么办

上面说的"跑一批数据校准 → 算 scale"属于 **PTQ（Post-Training Quantization，训练后量化）**。
它简单、不需要训练，但 **scale 是统计出来的，不一定最优**——遇到检测头这类对数值敏感的结构，
掉点会比较明显。

**TQT（Trained Quantization Thresholds，训练量化阈值）** 是 esp-ppq 内置的精度提升手段
（论文发表于 MLSys 2020）：把 scale 当作**可学习参数**，用少量校准数据做一次轻量微调。

| | PTQ | TQT |
|---|---|---|
| 需要标签 | 不需要 | **不需要**，校准集即可 |
| scale 怎么来 | 统计（minmax / KL / percentile） | 梯度下降学出来（仍约束为 2 的幂） |
| 优化目标 | 校准误差 | **浮点输出 vs 量化输出的 MSE** |
| 训练什么 | 无 | scale，可选权重（块级微调） |
| 耗时 | 分钟级 | 分钟～小时（看 steps / block_size / CPU 或 GPU） |
| 典型收益 | 基线 | 检测 mAP +1～3 个点，分类 Top-1 +1 个点左右 |

**TQT 怎么做：在 log 域学 scale**

esp-dl 要求 scale 必须是 `2^exponent`，所以 TQT 不直接学 scale，而是学它的对数：

```
alpha = log2(scale)          ← 可学习参数
scale = 2^alpha
```

- 这样天然满足 2 的幂约束，数值上也更稳定；
- 前向时用 `round(alpha)` 对应的整数 exponent（和硬件一致），反向用 **STE（直通估计）**
  把梯度传给 `alpha`；
- `int_lambda` 是正则项，把 `alpha` 往整数上拉，导出后和芯片端完全对齐；
- 损失是**浮点模型输出和量化模型输出的 MSE**，不需要标签，只需要校准集；
- 除了 scale，权重也可以一起参与块级微调（`is_scale_trainable` 控制）。

在 esp-ppq 里它实现为 `TrainedQuantizationThresholdPass`，不是独立的一步，而是
`espdl_quantize_onnx/torch` 内部的一个 pass，**打开开关即可**。

**在 ESP32 上怎么开**

要求 esp-ppq ≥ 1.2.7：

```python
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, ENABLE_CUDA_KERNEL

quant_setting = QuantizationSettingFactory.espdl_setting()
quant_setting.tqt_optimization = True

tqt = quant_setting.tqt_optimization_setting
tqt.steps = 500                  # 每个 block 的微调步数
tqt.lr = 1e-5                    # 学习率
tqt.block_size = 4               # 图分块大小：越大越快，过大可能不稳定
tqt.int_lambda = 0.25            # 让 exponent 更接近整数（可选）
tqt.collecting_device = "cuda"   # 有 GPU 时缓存校准数据，加速

with ENABLE_CUDA_KERNEL():
    graph = espdl_quantize_onnx(
        onnx_import_file="model.onnx",
        espdl_export_file="model.espdl",
        calib_dataloader=calib_loader,
        calib_steps=32,
        input_shape=[1, 3, 224, 224],
        target="esp32p4",
        num_of_bits=8,
        setting=quant_setting,
        device="cuda",
    )
```

常用参数（`TQTSetting`）：

| 参数 | 默认 | 作用 |
|---|---|---|
| `steps` | 500 | 每个 block 的微调步数 |
| `lr` | `1e-5` | 学习率 |
| `block_size` | 4 | 图分块大小；越小越稳、越慢 |
| `is_scale_trainable` | `True` | `True` 同时训 scale + 权重；`False` 只训权重 |
| `int_lambda` | 0.0 | 惩罚 `alpha` 偏离整数，范围 [0,1] |
| `gamma` | 0.0 | MSE 正则项 |
| `interested_layers` | `[]` | 只微调指定层；空 = 所有满足条件的 Conv/Gemm |
| `collecting_device` | `cpu` | 校准数据缓存设备，有 GPU 改 `cuda` |

**限制**

- 只对满足 **POWER_OF_2 + LINEAR + SYMMETRICAL** 的量化配置生效，不满足就跳过
  （esp-dl 的默认配置满足）；
- 需要校准数据和额外算力，不是"免费"的；
- 优化的是重建 MSE，不是任务指标——**仍要用验证集评估最终精度**；
- 不保证每个模型都涨点；`block_size` 过大会不稳定（建议先试 4）。

**效果（官方实测，ESP32-P4）**

| 模型 | PTQ | PTQ + TQT |
|---|---|---|
| YOLO26n 640（e2e mAP50-95） | 0.332 | **0.363** |
| YOLO26n 512（e2e mAP50-95） | 0.315 | **0.341** |
| MobileNetV2（Top-1） | 70.850% | **71.775%**（float 71.878%） |

**什么时候用**

1. 先跑 **PTQ**，用真实数据评估端到端指标（不是只看某一层的余弦相似度）；
2. 掉点可接受（分类 Top-1 掉 <1%，检测 mAP 掉 <2~3 个点）→ 停在 PTQ；
3. 不达标 → 打开 **TQT**（成本最低的一档），同时可以配合换校准算法（KL/percentile）、
   per-channel 权重、混合精度（w8a16）、AutoQuant；
4. 还可以和 **Weight Equalization** 叠加（官方示例 espdet_pico 就是这么用的）；
5. 仍不达标，再针对敏感层做混合精度等更细的调整。

> 一句话：**PTQ 是默认选项，TQT 是 esp-ppq 里性价比最高的精度补救手段**——
> 不需要标签、不用重训模型，只多花一点校准时间。

> 说明：TQT 是 esp-ppq / esp-dl 路线特有的能力；C5 的 TFLM 路线没有对应实现，
> 只能在 TFLite Converter 的 PTQ 选项里调（量化粒度、representative dataset 等）。

## 6. FlatBuffers：文件格式也是内存布局

`.tflite` 和 `.espdl` 都基于 **FlatBuffers**。理解它，就能理解为什么端侧加载模型几乎不需要"解析"。

**传统格式（Protobuf / JSON / pickle）的痛点**

- **硬盘上**：一串按语法排列的字节流。
- **内存里**：程序必须把它读进 RAM，再**解析**——分配变量、创建对象、字符串转数字，
  重新搭出一套数据结构。

**FlatBuffers 的做法：让两者变成同一份数据**

- 序列化时就按**内存对齐和偏移量**把数据排好；
- 文件里就是一块连续的二进制 Buffer，**它在 flash 里的样子和在 RAM 里的样子完全一致**；
- 运行时把 buffer 基地址交给程序，按记录的**偏移量**做指针跳转即可读字段，不需要解析成对象树。

> 类比：
> 普通格式（Protobuf）= 乐高积木装在盒子里，玩之前得倒出来按图纸拼好；
> FlatBuffers = 出厂就已经拼好的乐高，打开盒子直接玩，想看哪块看哪块。

**对 MCU 意味着什么**

- **省 RAM**：不需要先解析出一整套对象，也不一定需要把权重重新拷贝一遍；
- **省时间**：没有解析阶段，加载 ≈ 把指针指过去。

**"允许零拷贝"不等于"一定零拷贝"**：

- 模型编进 flash 的 rodata（TFLM 常见做法）时，运行时可以直接按偏移访问；
- esp-dl 在开启 `CONFIG_SPIRAM_RODATA` / `SPIRAM_XIP_FROM_PSRAM` 时，启动日志会提示
  `fbs model is copied to PSRAM` —— 模型被整体拷到 PSRAM（大模型/频繁访问时更稳）；
- 是否拷贝取决于"加载方式 + 存储位置 + 配置"，不是格式本身决定的。

## 7. 给初学者的常识校正

- **"模型多大"和"运行占多少内存"是两回事**：权重可以放 flash/PSRAM；
  运行时还要分配激活/中间张量，这部分往往比权重还大（检测模型尤其明显）。
- **推理速度不等于参数量**：耗时由**计算量**决定（输出尺寸 × 卷积核 × 通道数……），
  和参数量的关系没那么直接。
- **float32 模型能跑，但不代表高效**：不量化时内存/带宽都是 4 倍，MCU 上通常要走
  int8（P4 用 SIMD，C5 的 float 甚至由软浮点库模拟）。
- **模型不是"越准越好"**：端侧还要看 RAM、Flash、延迟、功耗。
- **模型文件不能跨运行时/跨芯片通用**：`.tflite` 给 TFLM，`.espdl` 给 esp-dl；
  量化目标（芯片）不同也不能混用。

## 8. 两条实现路线

| 维度 | TFLite Micro + esp-nn（C5） | esp-dl + esp-ppq（P4） |
|---|---|---|
| 模型格式 | `.tflite` | `.espdl` |
| 量化工具 | TFLite Converter | esp-ppq（KL/percentile、TQT、AutoQuant） |
| 运行时 | TFLM 解释器 + `tensor_arena` | esp-dl execution plan + 静态内存规划 |
| 算子 | esp-nn kernel + TFLM 参考实现 | esp-dl Module + 芯片 SIMD |
| 适合 | 通用 TFLite 生态、跨平台 | 检测/识别、深度调量化、吃 P4 SIMD |
| 笔记 | [esp32-c5 分支](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-c5) | [esp32-p4 分支](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-p4) |

> 两条路线的层级关系（TinyML / TFLM / esp_nn / esp-dl）在各分支第 1 章有更详细的说明。

---

## 相关文档

- **[neural-networks.md · 神经网络基础](https://github.com/yh-xh/ESP32-TinyML/blob/main/neural-networks.md)** ——
  神经元、激活函数、MLP/CNN/TCN（先建立网络结构的直觉）。
- **[esp32-p4 分支](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-p4)** ——
  esp-dl + esp-ppq：量化调优、C++ 部署、自定义算子、esp-who。
- **[esp32-c5 分支](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-c5)** ——
  esp-nn + TFLite Micro：手势识别实战、软浮点、arena。
- 回到 [仓库首页](https://github.com/yh-xh/ESP32-TinyML)。

> 建议顺序：neural-networks.md（网络结构）→ 本文（量化与部署）→ 对应平台分支（实战）。
