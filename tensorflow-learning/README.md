# 从 Keras 到 ESP32：端侧模型训练与部署路线

> 目标：把模型训练出来，转成 int8 `.tflite`，部署到 XIAO ESP32S3 的 TFLite Micro + esp-nn 上。
> 环境：`conda env tf-esp`，Python 3.11 / TensorFlow 2.15.1 / Keras 2.15。
> 配套：`concepts.md` 是概念地图，遇到抽象概念先查这里；`python-notes.md` 是 Python 错题本；部署参考见本分支 [README.md](../README.md) 和 [TensorFlow.md](../TensorFlow.md)。

---

## 路线怎么定

先把目标说清楚：你要的不是“学完 TensorFlow”，而是把模型放到 XIAO 上跑起来。路线应该从部署倒推，而不是按框架文档的目录从头刷。

Keras 是主线。模型结构、损失、优化器、训练、评估、保存，这些工作大部分在 Keras 完成。TensorFlow 不是另一门平行的课；它在你需要“把模型送出去”时才真正出现：

- 模型导出；
- TFLite 转换；
- int8 量化；
- shape、dtype、算子的排查；
- 板端加载失败时回头看模型。

这个顺序不要反。

先啃 TensorFlow 底层，很容易把时间花在 TF1 的 `Graph`、`Session`、分布式、TFX 上；这些和 ESP32 部署没有直接关系。只学 Keras，模型能训练，但到 `.tflite`、量化、TFLM 算子这里就会停住。**Keras 负责把模型做出来，TensorFlow 负责把模型送出去，TFLM/esp-nn 负责让模型在板子上跑。**

路线按“产物”组织，不按“知识点”组织。每一阶段结束都应该有能运行的东西；没有产物的学习，不算进度。

---

## 先看一个已经跑通的例子

你刚跑通的 sine 模型，就是这条路线的小样：

```text
10,000 个 (x, 0.5*sin(x)) 样本
  → Keras: 1 → 32 → 32 → 1，训练 300 epoch
  → 保存 .keras
  → TensorFlow: TFLiteConverter + 200 个点校准
  → int8 .tflite，3,592 字节
  → tf.lite.Interpreter 在 PC 验证，最大误差 0.0118
  → xxd 生成 model.cc
  → TFLM/esp-tflite-micro 编译进固件
  → XIAO 串口输出，最大误差 0.0247
```

训练几乎全是 Keras；TensorFlow 只在导出、转换、量化、验证时出现；最后上板是 TFLM 和 esp-nn。这条路线不是理论推演，是已经验证过的路径。

---

## 路线总览

| 阶段 | 你会做什么 | 大概时间 | 结束时应该有什么 |
|---|---|---:|---|
| 0 | 跑通环境，补最少的 Python | 2–3 天 | 能跑通第一个脚本 |
| 1 | 看懂 Tensor、shape、NumPy | 1 周 | 能排查 shape 错误 |
| 2 | 用 Keras 训练第一个模型 | 1–2 周 | 一个能保存的 `.keras` |
| 3 | 把训练做得像样 | 1–2 周 | 训练曲线 + 验证集 |
| 4 | 换 CNN / 1D CNN | 1–2 周 | 图像或 IMU 小模型 |
| 5 | 学 TensorFlow 导出 | 3–5 天 | SavedModel / `tf.function` |
| 6 | 转 int8 TFLite | 1 周 | PC 验证过的 `.tflite` |
| 7 | 上 XIAO | 1–2 周 | 串口能出结果 |
| 8 | 做自己的项目 | 2–4 周 | 端到端项目和实测数据 |
| 9 | 以后再说 | 持续 | QAT、剪枝、性能优化 |

按每周 6–10 小时算，大约 10–14 周。不要赶，阶段验收比进度重要。

---

## 阶段 0：先跑起来

先做一件事：

```bash
conda activate tf-esp
cd tensorflow-learning
python practice/01-first-run/first_run.py
```

`first_run.py` 只做三件事：导入 TensorFlow、创建两个张量、打印它们。你会看到 `shape` 和 `dtype`，这是后面所有模型的起点。

这一阶段需要认识的 Python：

- `import`、变量、`print`；
- 缩进；
- 列表、字典（先混个脸熟）；
- 函数 `def`；
- `ModuleNotFoundError`、`Traceback`。

不要把 Python 当一门课先学完。遇到 `ModuleNotFoundError`，先检查 `which python` 是不是 `tf-esp` 环境；遇到 `IndentationError`，检查空格；遇到 shape 报错，先 `print(x.shape)`。

**跑通标志**：能改一个数字、重新运行、解释 `shape` 和 `dtype`。

---

## 阶段 1：shape 是第一道坎

Keras 能帮你隐藏梯度，但隐藏不了 shape。

你至少要知道：

- `(N,)` 和 `(N, 1)` 不是一回事；
- 输入层 `Input((1,))` 对应每个样本 1 个特征；
- 图像是 `(H, W, C)`，批量是 `(N, H, W, C)`；
- `reshape` 改的是形状，不改数据顺序；
- 矩阵乘 `@` 和逐元素乘 `*` 不一样。

这一阶段不用学 `GradientTape`。用 `tf.constant`、`tf.reshape`、`tf.matmul` 和 NumPy 做几个小实验：

```python
import tensorflow as tf
import numpy as np

a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[5., 6.], [7., 8.]])
print(a + b)
print(a * b)
print(a @ b)

x = np.linspace(0, 1, 5).reshape(-1, 1)
print(x.shape)
```

**跑通标志**：看到 `(10000,)` 能想到“需要 reshape 成 `(10000,1)`”；看到 shape mismatch，知道先打印两边的 shape。

---

## 阶段 2：Keras 成为主工具

这一阶段开始，主要精力放在 Keras 上。

先复现 sine 模型：

```python
import numpy as np
import tensorflow as tf

x = np.linspace(0, 2*np.pi, 10000, dtype=np.float32).reshape(-1, 1)
y = (0.5 * np.sin(x)).astype(np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Input((1,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
model.fit(x, y, epochs=300, batch_size=64, verbose=0)

print(model.predict(x[:5], verbose=0).reshape(-1))
model.save("sine.keras")
```

你需要能解释：

- 为什么输入是 `(1,)`；
- 为什么中间用 ReLU；
- 为什么最后一层不加激活；
- 为什么回归用 MSE；
- epoch 和 batch 的区别；
- 32/32 和 16/16 有什么差别。

这一阶段不要碰 `GradientTape`。Keras 已经把前向、反向、更新都封装好了。你的任务是学会用它，而不是重写它。

**跑通标志**：能自己改层数、改训练轮数，观察误差变化，并保存/重新加载模型。

---

## 阶段 3：把训练做得像样

会 `fit` 不等于会训练。你需要知道：

- 训练集误差低，不代表实机好；
- 要有验证集；
- 过拟合是什么样；
- 什么时候用 EarlyStopping；
- 怎么保存验证集最好的模型；
- 怎么看训练曲线。

给 sine 模型加：

```python
history = model.fit(
    x, y,
    validation_split=0.2,
    epochs=300,
    batch_size=64,
    verbose=0,
)
```

然后画 `history.history["loss"]` 和 `val_loss`。如果训练 loss 一直降、验证 loss 开始升，就是过拟合的信号。

这一阶段做一个小对比：16/16 训练 100 轮，和 32/32 训练 300 轮，哪个误差更小。我们在 sine 上做过：前者 int8 最大误差约 0.094，后者降到 0.0118。模型容量和训练轮数不是玄学，是可以用数据比较的。

**跑通标志**：能解释“训练集 MAE 很低”为什么不等于“模型好”；能保存验证集最优模型。

---

## 阶段 4：换一个更像真实任务的模型

sine 只是练手。接下来选一个方向：

- 图像：`(96, 96, 1)` 的小 CNN；
- IMU：`(200, 3)` 的 1D CNN。

你需要学：

- `Conv2D`、`MaxPooling2D`、`GlobalAveragePooling2D`；
- `Conv1D`、`MaxPooling1D`；
- Functional API；
- `model.summary()` 看输入输出 shape 和参数量；
- 训练前怎么归一化、怎么 reshape。

不用一上来做 MobileNet/YOLO。先用几层 Conv + Pool 跑通，知道 `model.summary()` 里每一层 shape 怎么变，比抄一个大模型有用。

**跑通标志**：能把自己的数据整理成模型要求的 shape，训练不报错，能解释卷积和池化在做什么。

---

## 阶段 5：TensorFlow 开始登场

到这里，Keras 已经够你训练了。接下来要处理“模型怎么离开 PC”。

你需要知道：

- `.keras` 是 Keras 模型文件；
- SavedModel 是 TensorFlow 的部署格式；
- `tf.function` 会把 Python 函数追踪成图；
- TFLite Converter 需要一张图，不是 Python 对象；
- 推理图里没有 loss、优化器、Dropout。

这一阶段不需要手写训练循环，只需要理解：

```text
Keras 模型 → 追踪成图 → 导出/转换
```

**跑通标志**：能保存 `.keras` 和 SavedModel；能解释为什么 TFLite 转换需要图；能用 `@tf.function` 包一个简单推理函数。

---

## 阶段 6：把模型变成 int8

这是整条链路的分水岭。

你要做：

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
```

然后：

- 用 `tf.lite.Interpreter` 在 PC 上跑；
- 看 `input_details` / `output_details`；
- 理解 `scale` / `zero_point`；
- 用 Netron 看 `FULLY_CONNECTED` / `CONV_2D`；
- 比较 float32 和 int8 的误差；
- 检查模型输入/输出 shape 和 dtype。

这一阶段最容易出的问题：

- 忘了 representative dataset；
- 输入 dtype 和模型不匹配；
- 以为 int8 是训练 dtype；
- 转换后算子 TFLM 不支持；
- 量化后精度掉太多。

**跑通标志**：能独立把一个 Keras 模型转成 int8 `.tflite`，并在 PC 上验证输入输出和误差。

---

## 阶段 7：上 XIAO

上板不是“再点一次运行”，而是换了一套运行时：

- PC：TensorFlow / TFLite Interpreter；
- XIAO：TFLM + esp-tflite-micro + esp-nn。

你需要：

- `xxd` 生成 `model.cc`；
- `g_model` / `g_model_len`；
- `alignas(8) const`；
- `MicroInterpreter`、`MicroMutableOpResolver`、`tensor_arena`；
- `idf.py build/flash`；
- 串口验证。

这一阶段 Python 基本退场，换来的是 C++ 和 ESP-IDF。你不需要成为 C++ 专家，只要看懂模型数组、解释器初始化和串口输出。

**跑通标志**：能把一个新模型替换进 `tflite_hello`，编译、烧录、串口输出和 PC 对得上。

---

## 阶段 8：做自己的项目

到这一步，不要再做 sine 了。

选一个：

- 摄像头人形/人脸二分类；
- IMU 手势识别。

完整走一遍：

1. 采数据、标标签；
2. 分训练/验证/测试；
3. 训练；
4. 量化；
5. 上板；
6. 测误差、耗时、arena；
7. 记录失败案例。

**跑通标志**：模型是你自己的数据训练的；每一个预处理步骤你能解释；换一个场景还能复测。

---

## 阶段 9：以后再说

这些不是现在不学，而是现在学了用不上：

- QAT（量化感知训练）；
- 剪枝、聚类；
- 复杂 `tf.data`；
- TensorBoard；
- `GradientTape` 手写训练；
- TFLM 源码、自定义算子；
- ESP-DL + esp-ppq。

等你有一个能跑的项目，再针对瓶颈补。

---

## Python 怎么补

`python-notes.md` 里已经列了 18 个会在 TF/Keras 里遇到的概念。你不需要从 Python 教程第一页开始看。

优先掌握这 6 个：

1. 变量和 `print`；
2. 列表/字典；
3. 函数 `def`；
4. 类和对象（因为 `model` 是对象）；
5. `with` 和装饰器（因为 `tf.function`）；
6. 生成器 `yield`（因为 representative dataset）。

其他概念遇到再查、再记。你的 Python 笔记不是语法书，是错题本。

---

## 怎么判断自己真的在进步

不要用“看了多少小时”衡量。用这些检查点：

- [ ] 能不看教程写出一个 `Sequential` 模型；
- [ ] 能解释 `(N,)` 和 `(N,1)`；
- [ ] 能画出训练/验证曲线；
- [ ] 能解释过拟合；
- [ ] 能把模型转成 int8 并在 PC 验证；
- [ ] 能替换 `model.cc` 并烧进 XIAO；
- [ ] 能用自己的数据训练一个二分类或三分类模型；
- [ ] 能说清楚 PC、TFLite、TFLM 的分工。

每完成一个，再进入下一阶段。

---

## 不要做的事

- 不要先啃 TensorFlow 1.x 的 `Graph` / `Session`；
- 不要一开始学 TFX、TF Serving、TF.js、TF Agents；
- 不要手写 `GradientTape` 训练循环，除非 Keras 不够用；
- 不要收集一堆教程却不跑代码；
- 不要拿训练集准确率当实机效果；
- 不要一开始就做 YOLO/人脸识别大模型；
- 不要为了“学完 Python”停下来。

---

## 资料

- Python 官方教程：<https://docs.python.org/zh-cn/3/tutorial/>
- NumPy 快速入门：<https://numpy.org/doc/stable/user/quickstart.html>
- Keras 指南：<https://keras.io/guides/>
- TensorFlow 教程：<https://www.tensorflow.org/tutorials>
- 本仓库 [TensorFlow.md](https://github.com/yh-xh/ESP32-TinyML/blob/TFLite-micro/TensorFlow.md)
- TFLite 转换：<https://ai.google.dev/edge/litert/models/convert>
- 训练后量化：<https://ai.google.dev/edge/litert/models/post_training_quantization>
- TFLM 总览：<https://ai.google.dev/edge/litert/microcontrollers/overview>
- esp-tflite-micro：<https://github.com/espressif/esp-tflite-micro>
- esp-nn：<https://github.com/espressif/esp-nn>
- Netron：<https://netron.app/>

---

## 最后

这条路线不是让你“学完 TensorFlow”，而是让你在 10–14 周内拥有一个能上板的完整闭环。

Keras 负责把模型做出来，TensorFlow 负责把模型送出去，TFLM/esp-nn 负责让模型在 XIAO 上跑起来。你不需要成为三者中的任何一个专家，但要知道它们各自在什么时候出现。

先把 `practice/01-first-run/first_run.py` 跑通。跑通，再往下。
