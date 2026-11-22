# 概念地图：把零碎知识放回一张图

这份文档不是教材，也不是要你一次背完的清单。

它是给你查漏补缺用的：跑练习时遇到一个抽象概念，先来这里看一句话和它在哪一节出现过，再回去看代码。

你不需要现在把下面所有东西都看懂。先按练习顺序走，遇到不懂的再查。

如果你现在觉得乱，先只做三件事：

1. 跑通 03，能解释 loss、epoch、batch 是什么；
2. 跑通 04，能解释训练集/验证集/测试集和早停；
3. 跑通 05，能画出 CNN 每一层的 shape。

其他的先放着，遇到再查。

## 先分清优先级

| 优先级 | 内容 | 现在要做什么 |
|---|---|---|
| 现在必须 | tensor、shape、Dense、loss、optimizer、epoch/batch、训练/验证/测试、过拟合、卷积基本 shape | 能用自己的话解释，能算简单参数量 |
| 先混个脸熟 | logits、softmax、padding/stride、dilation、GAP、Sequential vs Functional | 知道它是干什么的，不要求推导 |
| 上板再说 | TFLite int8、scale/zero_point、TFLM arena、esp-nn | 到 06/07 再深入，现在知道名字即可 |

不要按字母顺序背。按你正在跑的练习查。

---

## 一张主线图

```text
数据
  -> 张量 shape / dtype
  -> 模型层（Dense / Conv / Pool）
  -> loss
  -> optimizer
  -> fit（前向、反向、更新）
  -> 验证集 / 测试集
  -> 导出 .keras
  -> TFLite 转换 / int8 量化
  -> TFLM + esp-nn 上板
```

你现在的进度在“模型层”和“验证集”之间，后面才是导出和上板。

---

## 0. TensorFlow、Keras、TFLite、TFLM 分别是什么

| 名字 | 位置 | 干什么 |
|---|---|---|
| TensorFlow | PC | 底层框架：张量、自动微分、计算图、保存和转换 |
| Keras | PC | 高层 API：`Sequential`、`Dense`、`fit`、`evaluate` |
| TFLite | PC / 手机 / MCU | 把模型转成 `.tflite`，提供轻量推理 |
| TFLM | MCU | TensorFlow Lite Micro，单片机上的推理运行时 |
| esp-nn | ESP32 | 乐鑫的算子加速库，TFLM 在 ESP32 上会用它 |

你现在写的：

```python
tf.keras.Sequential(...)
model.fit(...)
```

是 Keras 的写法；`tf.keras` 表示它是 TensorFlow 里的 Keras。

`import tensorflow` 每次都从本地磁盘加载库、初始化运行时，不访问服务器；`pip install` 只是把文件装到本地，装完和每次导入是两件事。

参考：[README.md](README.md)

---

## 1. 张量和 shape

### tensor

TensorFlow 里的多维数组。它比 NumPy 数组多了 dtype、设备、自动微分等能力。

```python
tf.constant([1.0, 2.0, 3.0])   # shape (3,), dtype float32
```

### shape

每一维多大。

```text
()        标量
(3,)      长度为 3 的向量
(3, 1)    3 行 1 列
(2, 3)    2 行 3 列
(N, 16, 16, 1)   batch、高、宽、通道
```

### dtype

元素类型。模型常用 `float32`；量化后的 TFLite 常用 `int8`。

### ndim / rank

有几个维度。标量 0，向量 1，矩阵 2。

### batch 维

模型输入通常第一维是 batch：

```text
(N, features)
```

`N` 是这一批样本数。`Input((1,))` 不包含 batch，实际输入是 `(N,1)`。

`model.summary()` 里的 `None` 就是 batch 不固定。

### reshape

改形状，不改数据顺序。元素总数必须一致；`-1` 表示自动推断。

### 广播

shape 不一样时，按规则虚拟扩展：

- 从最右边对齐；
- 相等，或其中一个是 1，才能算；
- 缺维度当成 1；
- 输出取每维较大值。

### `@` 和 `*`

```text
@  矩阵乘：(N,K) @ (K,M) -> (N,M)
*  逐元素乘：对应位置相乘
```

参考：[01-first-run](practice/01-first-run/README.md)、[02-tensor-shapes](practice/02-tensor-shapes/README.md)

---

## 2. 模型、层、参数

### layer

模型的一层。比如：

```text
Dense        全连接
Conv2D       二维卷积
MaxPooling2D 二维最大池化
Flatten      拉平
```

### weight / bias

层里可训练的数。Dense 和 Conv 都有 W 和 b。

### parameter

模型里所有可训练的数。`model.summary()` 的 `Total params` 就是它。

### hyperparameter

人定的，不是训练学出来的：

```text
层数、每层 units/filters、卷积核大小、学习率、batch size、patience
```

### Dense 的 shape

```text
W shape = (输入维度, 神经元数)
b shape = (神经元数,)
参数量  = 输入维度 × 神经元数 + 神经元数
```

### 激活函数

| 函数 | 输出范围 | 常用在哪 |
|---|---|---|
| linear | 任意实数 | 回归最后一层 |
| ReLU | `max(0, z)` | 隐藏层 |
| sigmoid | `0~1` | 二分类 / 多标签 |
| softmax | 每类概率，和为 1 | 多分类最后一层 |

### logits

softmax/sigmoid 之前的原始分数。可正可负，不是概率。

### Sequential 和 Functional

```text
Sequential   一条直线，单输入单输出
Functional   有分支、多输入/多输出、共享层
```

`Sequential` 的名字是“层按顺序堆叠”，不是“处理序列数据”。IMU 时间序列也可以用它，只要模型是一条直线。

### 模型不是公式

模型没有把 `cos` 公式写进去。它是用权重和激活函数拼出一条近似曲线，只在训练覆盖的范围内比较可靠；喂区间外的值，可能很离谱。

### 参数什么时候会变多

固定宽度加深，参数是线性增加；真正容易让参数爆炸的是宽度和输入维度。比如把 `96×96` 图像拉平接 Dense，一层就能到上百万参数。

参考：[03-first-keras-model](practice/03-first-keras-model/README.md)、[05-cnn](practice/05-cnn/README.md)

---

## 3. 训练到底在干什么

### 前向

输入过模型，得到预测。

### loss

预测和目标差多少。

```text
MSE                    回归
binary_crossentropy    二分类
categorical_crossentropy        多分类，one-hot 标签
sparse_categorical_crossentropy 多分类，整数标签
```

### 梯度 / 反向传播

算每个参数应该往哪个方向调、调多少。

### optimizer

根据梯度更新参数。Adam 是常用优化器。

### 学习率

每次更新的步长。太大震荡，太小慢。

### epoch / batch / step

```text
epoch   把全部训练数据过一遍
batch   一次拿多少个样本更新
step    一次参数更新，也就是处理一个 batch
```

```text
steps_per_epoch = ceil(样本数 / batch_size)
总更新次数      = steps_per_epoch × epochs
```

不是每个 batch 训练一个模型再合并。始终是同一个模型：用当前 batch 算梯度、更新一次参数，下一批从更新后的参数继续。最后的模型就是这组参数经过所有更新后的结果，不是多个模型平均。

### 训练策略不止一种

上面说的是标准的 mini-batch 梯度下降，也是 `model.fit` 默认的做法。但“训练模型”不只这一种：

| 策略 | 做法 |
|---|---|
| full-batch | 一次用全部数据算梯度，每个 epoch 更新一次 |
| SGD | batch=1，每个样本更新一次 |
| mini-batch | 一次用一批，最常用 |
| 梯度累积 | 多个小 batch 累加梯度，再更新一次，模拟大 batch |
| 数据并行 | 多卡各算一部分梯度，梯度求平均后更新同一个模型 |
| 集成 | 独立训练多个模型，预测时平均概率或投票 |
| 权重平均 | SWA / EMA / model soup，把多轮权重平均成一个模型 |
| 联邦学习 | 多个客户端本地训练，服务器聚合权重 |

注意区分：

- 数据并行合并的是**梯度**，还是一个模型；
- 集成保留多个模型，合并的是**预测**；
- 权重平均合并的是**权重**，最后得到一个模型。

`model.fit` 默认走的是单模型 + mini-batch；其他策略需要自定义训练循环、分布式配置或额外代码。

### shuffle

`fit` 默认每个 epoch 打乱训练数据。顺序不同，更新路径会不同。

### history

`fit` 返回的对象，里面存每轮的 loss / accuracy / val_loss / val_accuracy。

### verbose

```text
0  不打印
1  进度条
2  每个 epoch 一行
```

### 随机种子

固定初始化、数据打乱等随机过程：

```python
tf.keras.utils.set_random_seed(42)
```

### 动态图和静态图

```text
动态图 / eager：Python 执行一步算一步，调试直观
静态图 / graph：先把计算过程固定成图，再整体执行/优化
```

`tf.function` 会把 Python 函数追踪成图。Keras 的 `fit` 内部会用图执行训练步骤，所以你不用自己写 `Session`。TFLite 转换也需要一张图，而不是一个 Python 对象。

参考：[03-first-keras-model](practice/03-first-keras-model/README.md)

---

## 4. 评估、过拟合、早停

### 训练集 / 验证集 / 测试集

```text
训练集：更新参数
验证集：每个 epoch 算 val_loss，参与早停和模型选择
测试集：训练和调参都不看，最后评估一次
```

### 过拟合

训练 loss 继续降，验证 loss 不再改善甚至上升。模型记住了训练数据，没有学到泛化规律。

### 欠拟合

训练 loss 和验证 loss 都高。模型容量不够、训练不够、特征不够。

### 泛化

模型在没见过的数据上的表现。

### 正则化

“正则化”不是正则表达式。它是一类**给模型加约束、让它别把训练集背得太死**的方法。

数学上最常见的是在 loss 后面加惩罚项：

```text
L_total = L_data + λ × R(W)
```

- `L_data`：原来的任务 loss，比如 MSE；
- `R(W)`：对权重的惩罚；
- `λ`：惩罚强度，越大约束越强。

常见形式：

| 方法 | 做什么 |
|---|---|
| L2 / weight decay | 惩罚大权重，让权重更小、更平滑 |
| L1 | 惩罚绝对值，容易把一部分权重压到 0，可用于稀疏/压缩 |
| Dropout | 训练时随机丢弃一部分神经元，防止过度依赖固定组合 |
| 数据增强 | 翻转、裁剪、加噪，让模型见到更多变化 |
| EarlyStopping | 验证 loss 变差就停，限制训练走得太远 |
| Label smoothing | 不把标签当成绝对 0/1，降低过度自信 |
| 权重共享 / 更小的模型 | 直接减少模型容量，CNN 本身就是一种结构约束 |

所以“正则化专门解决过拟合”是一句有用的简化，但不完全准确：

- 主要目的确实是提升泛化、抑制过拟合；
- 但 L1 还常用来做稀疏和模型压缩，这对单片机很有意义；
- L2 也能改善数值条件、限制权重过大；
- 数据增强还在教模型“哪些变化不应该改变答案”；
- EarlyStopping 更像训练控制，但常被归到正则化一类。

用过头会欠拟合：训练 loss 和验证 loss 都变差。先让模型有能力过拟合，再逐步加正则化，看验证集是否改善。

Keras 里常见写法：

```python
tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
tf.keras.layers.Dropout(0.2)
```

你已经在用其中一种：04 的 EarlyStopping。05 的 CNN 权重共享也算是结构上的约束，只是它没有写成 loss 惩罚项。

### EarlyStopping

监控 `val_loss`，连续 `patience` 轮没改善就停。

```text
monitor="val_loss"
patience=50
restore_best_weights=True
```

### ModelCheckpoint

把验证 loss 最好的模型保存到磁盘：

```python
ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True)
```

### 早停恢复权重算不算用验证集调参

算“模型选择”，不算梯度更新。

它没有拿验证集做反向传播，但它在多个 checkpoint 里挑了验证 loss 最低的那个。所以验证 loss 会略微偏乐观，测试集才保持独立。

### 测试集泄漏

反复用测试集调参、选模型，测试集就不再是“没见过的数据”。

### baseline

一个简单的对照，比如“永远预测训练集平均值”。模型至少要明显好于它。

### 混淆矩阵

行是真实类别，列是预测类别。对角线是判对的。

参考：[04-train-and-evaluate](practice/04-train-and-evaluate/README.md)

---

## 5. 任务和输出形式

### 回归

输出连续值，比如 `cos(x)`、坐标、距离。

### 二分类

输出一个概率，比如“有人/没人”。

### 多分类

输出多个类别概率，比如“x/y/z 轴”。

### 多标签

一张图可以同时有多个标签，每一类独立 sigmoid，不要求和为 1。

### 输出形式

```text
连续值      回归
logits      未归一化分数
概率        sigmoid / softmax 之后
类别标签    阈值 / argmax 之后
one-hot     人为转换的标签格式
检测框      坐标 + 类别分数
分割掩码    (N,H,W,C) 每个像素的类别
embedding   (N,d) 特征向量
```

参考：[05-cnn](practice/05-cnn/README.md)

---

## 6. 卷积

### 为什么用卷积

- 局部连接：每个输出只看一小块；
- 权重共享：同一个卷积核在整张图/整段时间上复用；
- 参数不随图像尺寸爆炸。

### Conv2D / Conv1D 的 shape

```text
Conv2D: (N, H, W, C_in)  -> (N, H_out, W_out, filters)
Conv1D: (N, L, C_in)     -> (N, L_out, filters)
```

### kernel / filters / channels

```text
kernel_size  卷积核看多大，人定的超参数
filters      输出通道数，也就是几个卷积核
channels     输入/输出的特征图数量
```

`kernel_size` 是你搭模型时定的，训练只更新卷积核里的数值，不会改变核的大小。

Keras 卷积核实际 shape：

```text
Conv2D: (kernel_h, kernel_w, C_in, filters)
Conv1D: (kernel_size, C_in, filters)
```

### 参数量

```text
Conv2D: kernel_h × kernel_w × C_in × filters + filters
Conv1D: kernel_size × C_in × filters + filters
```

### padding

```text
same   补零，stride=1 时 H/W 不变
valid  不补，3×3 会让 H/W 各减 2
```

`same` 只保证 H/W 不变，通道数会变成 `filters`。

padding 是卷积层自己的参数，不是“给池化用的”；池化也有自己的 padding，默认 `"valid"`。

空间尺寸由输入大小、卷积核、padding、stride、dilation 共同决定，不是只看 stride。

### stride

卷积核每次移动几格。

```text
stride=1  每个位置都算
stride=2  隔一个位置算，输出约减半
```

### 输出尺寸

默认 dilation=1：

```text
H_out = floor((H_in + 2×padding - kernel) / stride) + 1
```

`same` 且 stride=1 时，`H_out = H_in`。

### 池化

`MaxPooling2D` / `MaxPooling1D` 取窗口最大值。

```text
默认 strides = pool_size
```

所以 `MaxPooling2D((2,2))` 默认 stride 2，高宽减半；没有可训练参数。

### Flatten / GAP

```text
Flatten    保留所有位置，拉成一维，后面 Dense 参数多
GAP        每个通道求平均，参数少，更抗位置变化
```

### dilation

卷积核元素之间隔开，扩大感受野，不增加参数量。

```text
effective_kernel = (kernel_size - 1) × dilation + 1
```

适合“想要大感受野，但不想降分辨率”的场景，比如分割、长序列。

### 感受野

一个输出位置能看到多大的输入区域。堆卷积、池化、dilation 都会扩大感受野。

### CNN vs Dense

```text
96×96 拉平接 Dense(128)：约 118 万参数
Conv2D(32, 3×3) 输入 1 通道：320 参数
```

参考：[05-cnn](practice/05-cnn/README.md)

---

## 7. 部署相关

### `.keras` 和 `.tflite`

```text
.keras    训练格式：结构 + 权重 + 优化器状态
.tflite   推理格式：只保留推理算子，通常是 int8
C 数组    .tflite 转成 model.cc，编译进固件
```

### int8 量化

把 float32 权重和激活压成 int8，减少体积和内存。

需要 representative dataset 校准。

### scale / zero_point

int8 和 float 的换算：

```text
float = (int8 - zero_point) × scale
```

### TFLM / esp-nn

TFLM 在单片机上加载模型、分配 tensor arena、执行算子；esp-nn 加速其中的 int8 算子。

参考：[README.md](README.md)、[本分支的部署笔记](../README.md)

---

## 8. 最容易混淆的对照表

| 容易混 | 区别 |
|---|---|
| epoch / batch / step | 数据过一遍 / 一次拿多少 / 更新一次 |
| parameter / hyperparameter | 训练学出来的 / 人定的 |
| loss / metric | 参与梯度 / 只用于观察 |
| logits / probability / label | 原始分数 / 归一化概率 / 最终类别 |
| 验证集 / 测试集 | 参与选择 / 最后只看一次 |
| padding / stride | 边缘补零 / 移动步长 |
| kernel_size / filters | 看多大 / 输出几个通道 |
| Flatten / GAP | 保留位置 / 每通道求平均 |
| Sequential / Functional | 直线 / 任意分支 |
| `.keras` / `.tflite` | 训练格式 / 推理格式 |
| `same` / `valid` | 保持 H/W / 不补零 |
| Conv1D / Conv2D | 时间一维 / 图像高宽二维 |
| `fit` / `evaluate` / `predict` | 训练 / 评估 / 推理 |
| `verbose=0/1/2` | 静默 / 进度条 / 每轮一行 |

---

## 9. 怎么学，才不会被概念淹没

### 一次只跟一个练习

不要同时开 03、04、05。跑通一个，改两个参数，再看下一个。

### 先跑，再看，再改

```text
跑通 -> 看输出 -> 改一行 -> 再跑
```

比先看完所有理论有效。

### 遇到抽象概念，先找具体 shape

比如“什么是卷积”，不要先背定义，先画：

```text
16×16×1 -> Conv2D(8,3×3) -> 16×16×8 -> Pool -> 8×8×8
```

shape 对上了，概念就具体了。

### 不背 API，记三件事

```text
输入 shape 是什么
输出 shape 是什么
参数怎么算
```

API 名字可以查，shape 和公式要熟。

### 用这份文档当索引

不懂 `dilation`，来这里看一句话，再回 05 看代码和注释；不要从头读到尾。

### 记录报错

把 shape mismatch、dtype mismatch 的报错和原因记到 `python-notes.md` 或自己的笔记里。下次遇到会快很多。

### 现在不用懂所有东西

下面这些现在只要知道名字：

```text
logits、dilation、int8 量化、scale/zero_point、tensor arena、esp-nn 加速
```

到 06/07 会自然用到，到时候再深入。

---

## 最后

概念不是靠一次背完的，是靠反复用出来的。

你现在能跑通 `Dense`、CNN、验证集和早停，已经比自己以为的走得远。剩下的，遇到一个查一个，别急着一次全懂。
