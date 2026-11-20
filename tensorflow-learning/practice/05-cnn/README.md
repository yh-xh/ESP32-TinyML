# 卷积：让模型看局部，而不是把数据拉平

前面的 Dense 层把每个输入和每个输出都连起来。对 `cos(x)` 这种一维输入没问题，但图像和时间序列不行：

- 图像相邻像素有关系，拉平以后位置信息丢了；
- 全连接参数量随输入尺寸暴涨；
- 同一个图案出现在不同位置，Dense 要分别学，卷积可以共用一套权重。

这一节做两个任务：

```text
image_cnn.py   16×16 灰度图，分辨横条 / 竖条（二分类）
imu_cnn.py     200×3 振动信号，分辨 x / y / z 轴（三分类）
```

第二个任务的输入 shape 和 C5 上的 IMU 手势数据一致：`200 × 3`。

代码里每个 API、每个 shape 变化都有注释；这份文档讲卷积的思路、参数怎么算、结果怎么看。

## 先跑

```bash
conda activate tf-esp
cd practice/05-cnn
python image_cnn.py
python imu_cnn.py
```

会生成：

```text
image_cnn.keras    image_samples.png    image_training.png
imu_cnn.keras      imu_samples.png      imu_training.png
```

两个脚本各十几秒，数据是合成的，不依赖下载。

---

# 一、图像 CNN：横条还是竖条

## 数据

每张图 16×16，单通道灰度：

```text
(N, 16, 16, 1)
 │   │   │   └─ 通道数，灰度是 1，RGB 是 3
 │   │   └───── 宽
 │   └───────── 高
 └───────────── batch
```

横条是随机选几行涂白，竖条是随机选几列涂白，厚度 2～3 像素。数据本身是 0 背景、1 条；图里为了看得清，用了反向灰度显示，所以黑条代表 1，白底代表 0。

一共 1000 张，按 800 / 100 / 100 切成训练、验证、测试。

## 模型 shape 账本

```text
N × 16 × 16 × 1
  │ Conv2D(8, 3×3, same)
  ▼
N × 16 × 16 × 8
  │ MaxPooling2D(2×2)
  ▼
N × 8 × 8 × 8
  │ Conv2D(16, 3×3, same)
  ▼
N × 8 × 8 × 16
  │ MaxPooling2D(2×2)
  ▼
N × 4 × 4 × 16
  │ Flatten
  ▼
N × 256
  │ Dense(1, sigmoid)
  ▼
N × 1
```

| 层 | 输出 shape | 参数量 | 参数量怎么来 |
|---|---|---:|---|
| `Conv2D(8, 3×3)` | `(N,16,16,8)` | 80 | `3*3*1*8 + 8` |
| `MaxPooling2D(2×2)` | `(N,8,8,8)` | 0 | 没有可训练参数 |
| `Conv2D(16, 3×3)` | `(N,8,8,16)` | 1,168 | `3*3*8*16 + 16` |
| `MaxPooling2D(2×2)` | `(N,4,4,16)` | 0 | 没有可训练参数 |
| `Flatten` | `(N,256)` | 0 | `4*4*16 = 256` |
| `Dense(1, sigmoid)` | `(N,1)` | 257 | `256*1 + 1` |

总参数：

```text
80 + 1168 + 257 = 1505
```

约 5.88 KB（float32）。

Keras 里卷积核的实际 shape 是：

```text
(核高, 核宽, 输入通道, 输出通道)
```

所以：

```text
Conv2D(8, (3,3))  输入 1 通道  -> kernel (3, 3, 1, 8)
Conv2D(16, (3,3)) 输入 8 通道  -> kernel (3, 3, 8, 16)
```

输出通道数由 `filters` 决定；输入通道数由上一层决定。

## 卷积核大小是怎么定的

`(3,3)` 和 `8` 都是人定的超参数，不是训练学出来的。

- `(3,3)`：卷积核大小，决定一次看多大的局部区域；
- `8`：输出通道数，也就是 8 个不同的卷积核；
- 训练只更新卷积核里的数值和偏置。

常用选择：

| 卷积核 | 用途 |
|---|---|
| 1×1 | 只看单个位置，主要做通道组合 / 降维 |
| 3×3 | 图像最常用的起点，参数少，堆两层能扩大感受野 |
| 5×5 / 7×7 | 更大的图案，参数更多 |
| 9 / 11（1D） | 时间序列里看更长的一段波形 |

图像里 3×3 特别常见，因为它是能同时看到上下左右和中心的最小奇数核；两层 3×3 叠起来，感受野相当于 5×5，但参数更少，而且中间多了一次 ReLU。

没有公式能直接算出“这里必须用 3×3”。核越大，感受野越大，参数和计算也越多。先用 3×3，效果不够再试 5×5，或者叠两层；最终用验证集比较。

### dilation：不增加参数，扩大感受野

`dilation_rate` 让卷积核的元素之间隔开，不再紧挨着。

一维、卷积核大小 3 的例子：

```text
dilation=1:  [x x x]          连续 3 个点
dilation=2:  [x . x . x]      隔一个点，覆盖 5 个位置
dilation=3:  [x . . x . . x]  隔两个点，覆盖 7 个位置
```

等效卷积核大小：

```text
effective_kernel = (kernel_size - 1) * dilation + 1
```

所以 3×3、dilation=2 的等效感受野是 5×5，但参数仍然只有 9 个权重（乘以通道数），不是 25 个。

它解决的问题是：

```text
想要更大的感受野，但不想用大卷积核，也不想用池化/stride 降分辨率
```

典型使用场景：

- 语义分割：需要看很大范围，但输出要保留每个像素的位置；
- 时间序列 / 音频：想看到更长的波形，但不想把时间轴压得太短；
- 需要长距离上下文，又不想堆很多层或加池化。

代价：

- 采样是稀疏的，可能漏掉细小的局部模式；
- 多个 dilation 叠加可能出现“网格效应”；
- 理论计算量不变，但实际速度可能受内存访问方式影响；
- 上板前要确认 TFLM / esp-nn 对 dilation 的支持和性能。

Keras 写法：

```python
tf.keras.layers.Conv1D(16, 3, padding="same", dilation_rate=2)
tf.keras.layers.Conv2D(8, (3, 3), padding="same", dilation_rate=2)
```

注意：Keras 里 `strides > 1` 和 `dilation_rate > 1` 通常不要同时设置，二选一。你现在的模型都用默认 `dilation_rate=1`，所以不用改。

## padding 和池化的 shape

`padding="same"`：在边缘补零，stride=1 时输出高宽和输入一样。

`padding="valid"`：不补零，3×3 卷积会让高宽各减 2。

注意：`same` 只保证 **H/W（1D 是时间长度）** 一致，通道数会变成 `filters`：

```text
(N, 16, 16, 1) --Conv2D(8, same)--> (N, 16, 16, 8)
```

只有 `filters` 恰好等于输入通道数时，完整 shape 才完全一样。

默认步幅：

```text
Conv2D / Conv1D：strides=1
MaxPooling2D / MaxPooling1D：strides=None，实际等于 pool_size
```

注意池化的默认步幅不是 1，而是池化窗口大小。`MaxPooling2D((2,2))` 默认 stride 是 `(2,2)`，`MaxPooling1D(2)` 默认 stride 是 2。

另外，卷积的 `padding` 默认是 `"valid"`，所以代码里才显式写 `"same"`。

注意：`padding` 是**卷积层自己的参数**，不是“上一层输出给池化用的参数”。它决定卷积输出多大；池化只是接收卷积的输出。池化自己也有 `padding` 参数，默认是 `"valid"`。

```text
16×16 --Conv2D 3×3, same--> 16×16
16×16 --Conv2D 3×3, valid-> 14×14
```

通用公式（stride=1）：

```text
卷积输出 = floor((输入 + 2×padding - 卷积核) / stride) + 1
池化输出 = floor((输入 - 池化窗口) / stride) + 1
```

以 16×16 为例：

```text
Conv2D 3×3, same:      (16 + 2 - 3) + 1 = 16
MaxPool 2, stride 2:   (16 - 2) / 2 + 1 = 8
```

`stride` 是卷积核每次移动的格数。stride=1 时，卷积核依次停在 0、1、2、…、15，输出 16 个位置；stride=2 时，它只停在 0、2、4、…、14，输出 8 个位置，所以空间尺寸大约减半。

```text
stride=1: 位置 0 1 2 3 4 ... 15  -> 16 个输出
stride=2: 位置 0   2   4   ... 14 -> 8 个输出
```

二维图像高、宽两个方向都这样做，所以 `16×16` 会变成 `8×8`，输出位置数变成原来的 1/4。

注意“减半”是近似说法：`same` 时输出是 `ceil(输入 / stride)`，`valid` 时还要看卷积核大小和是否能放下。比如 `16×16`、`3×3`、`stride=2`、`padding="valid"`，输出是 7×7，不是 8×8。

所以空间尺寸不是只由 stride 决定。输入大小、卷积核大小、padding、dilation 都会影响输出；池化、GlobalAveragePooling、上采样也能改变空间尺寸。stride 只是其中最直接的下采样旋钮之一。

池化没有可训练参数，只负责减少分辨率、保留最强响应。

## 训练和结果

```text
优化器：Adam(1e-3)
loss：binary_crossentropy
metric：accuracy
20 个 epoch，batch 32
EarlyStopping 监控 val_loss，patience=5
```

这次验证 loss 一直在改善，EarlyStopping 没有触发，20 轮全部跑完。它不是必须触发，只是防止验证 loss 变差时继续浪费时间。

实际结果：

```text
test loss: 0.0004370563547126949
test accuracy: 1.0

confusion matrix (rows=true, cols=pred):
[[38  0]
 [ 0 62]]
```

测试集 100 张，38 张横条、62 张竖条，全部判对。

`predict` 返回的是概率，不是类别。要得到类别，自己做阈值：

```python
pred = (prob >= 0.5).astype(np.int32)
```

这个任务很简单，100% 不代表真实图像任务也这么容易。它只是让你看清 CNN 的 shape 和训练流程。

打开 `image_samples.png` 看测试图，打开 `image_training.png` 看 loss / accuracy 曲线。

---

# 二、IMU 1D CNN：x、y 还是 z

## 数据

每条信号 200 个时间点，3 个轴：

```text
(N, 200, 3)
 │    │    └─ 通道数：x / y / z
 │    └────── 时间步
 └─────────── batch
```

生成方式：

- 随机选一个轴；
- 在该轴上放一个两周期的正弦波；
- 相位、幅度随机；
- 三个轴都加一点噪声。

所以模型要判断“主要振动出现在哪个轴”。

一共 600 条，按 480 / 60 / 60 切分。这个 `200 × 3` 就是 C5 手势项目里 IMU 数据的 shape。

## 模型 shape 账本

```text
N × 200 × 3
  │ Conv1D(16, 9, same)
  ▼
N × 200 × 16
  │ MaxPooling1D(2)
  ▼
N × 100 × 16
  │ Conv1D(32, 5, same)
  ▼
N × 100 × 32
  │ GlobalAveragePooling1D
  ▼
N × 32
  │ Dense(3, softmax)
  ▼
N × 3
```

| 层 | 输出 shape | 参数量 | 参数量怎么来 |
|---|---|---:|---|
| `Conv1D(16, 9)` | `(N,200,16)` | 448 | `9*3*16 + 16` |
| `MaxPooling1D(2)` | `(N,100,16)` | 0 | 没有可训练参数 |
| `Conv1D(32, 5)` | `(N,100,32)` | 2,592 | `5*16*32 + 32` |
| `GlobalAveragePooling1D` | `(N,32)` | 0 | 没有可训练参数 |
| `Dense(3, softmax)` | `(N,3)` | 99 | `32*3 + 3` |

总参数：

```text
448 + 2592 + 99 = 3139
```

约 12.26 KB。

卷积核 shape：

```text
Conv1D(16, 9) 输入 3 通道  -> kernel (9, 3, 16)
Conv1D(32, 5) 输入 16 通道 -> kernel (5, 16, 32)
```

## 1D 卷积在干什么

`Conv1D` 的卷积核只沿时间轴滑动，输入通道是 3 个轴：

```text
取连续 9 个时间点 × 3 个轴
和 9×3 的卷积核逐元素相乘，求和
加偏置，过 ReLU
```

它学的是时间上的局部模式：上升沿、下降沿、周期性抖动。

这里的 `9` 和 `5` 也是人定的：

- 9 表示一次看 9 个连续采样点；
- 5 表示在池化后的时间轴上再看 5 个点，折算回原始数据大约覆盖 10 个点；
- 核越长，看到的时间范围越大，参数也越多。

长度公式和 `Conv2D` 一样，只是只有时间一维：

```text
L_out = floor((L_in + 2*pad - k) / stride) + 1
```

`padding="same"` 且 stride=1 时，`L_out = L_in`；池化 2、stride=2 时，长度减半。

## 为什么用 GlobalAveragePooling1D

第二层卷积后是 `(100, 32)`。两种接法：

```python
Flatten()                    # (100,32) -> 3200
Dense(3)                     # 3200*3+3 = 9603 参数
```

或者：

```python
GlobalAveragePooling1D()     # 每个通道对 100 个时间点求平均 -> 32
Dense(3)                     # 32*3+3 = 99 参数
```

GAP 把每个通道压缩成一个数，参数少很多，也不要求时间长度完全固定。代价是丢掉具体时间位置，只保留“这个通道整体有多强”。对“哪个轴在振动”这个任务足够。

## 训练和结果

```text
优化器：Adam(1e-3)
loss：sparse_categorical_crossentropy
metric：accuracy
30 个 epoch，batch 32
EarlyStopping 监控 val_loss，patience=5
```

实际结果：

```text
test loss: 0.0012919462751597166
test accuracy: 1.0

confusion matrix (rows=true, cols=pred):
[[14  0  0]
 [ 0 17  0]
 [ 0  0 29]]
```

测试集 60 条，14 条 x、17 条 y、29 条 z，全部判对。

`predict` 返回 `(60,3)` 的概率矩阵。要得到类别：

```python
pred = np.argmax(probs, axis=1)
```

打开 `imu_samples.png` 看三类信号，打开 `imu_training.png` 看训练曲线。

---

# 三、CNN 为什么比 Dense 适合图像

以 16×16 灰度图为例：

```text
拉平成 256 维，接 Dense(32)：
256 × 32 + 32 = 8,224 参数
```

上面的 CNN 只有 1,505 个参数。图像越大，差距越明显：

```text
96×96 拉平 = 9,216 维
接 Dense(128)：9,216 × 128 + 128 = 1,179,776 参数
```

一个卷积层：

```text
Conv2D(32, 3×3)，输入 1 通道：
3 × 3 × 1 × 32 + 32 = 320 参数
```

和图像是 16×16 还是 96×96 无关。

原因是：

- **局部连接**：每个输出只看一小块输入；
- **权重共享**：同一个卷积核在整张图上复用；
- **层次组合**：第一层学边缘，第二层组合成形状，更深层组合成物体。

这也是为什么后面做人形/人脸检测时，不能把摄像头图像直接拉平接 Dense。

---

# 四、输出怎么读

## 二分类

```python
Dense(1, activation="sigmoid")
loss="binary_crossentropy"
```

输出一个概率，通常以 0.5 为阈值。

## 三分类

```python
Dense(3, activation="softmax")
loss="sparse_categorical_crossentropy"
```

输出三个概率，取最大值的下标。

`model.predict` 返回概率，不返回类别：

```python
prob = model.predict(x_test, verbose=0)
pred = np.argmax(prob, axis=1)
```

## 混淆矩阵

行是真实类别，列是预测类别。对角线是判对的，非对角线是判错的。

```text
[[14  0  0]
 [ 0 17  0]
 [ 0  0 29]]
```

准确率只有一个数，混淆矩阵能看出“哪一类容易被判成哪一类”。真实项目里，后者更有用。

---

# 五、常见坑

### 通道维不能丢

```text
图像： (N, H, W, C)     灰度 C=1，RGB C=3
IMU：  (N, time, C)     C=3 个轴
```

`Conv2D` 要求输入是 `(H, W, C)`，不是 `(C, H, W)`。Keras 默认 `channels_last`。

### `padding` 会改变 shape

```text
padding="same"   输出高宽和输入一样（stride=1）
padding="valid"  不补零，3×3 卷积会让高宽各减 2
```

shape 报错时，先看每一层的输出 shape。

### 池化没有参数

`MaxPooling2D` / `MaxPooling1D` 只是取最大值，没有可训练权重。`model.summary()` 里 Param # 是 0。

### Flatten 和 GAP 的区别

- `Flatten`：保留所有位置，后面 Dense 参数多；
- `GlobalAveragePooling`：每个通道求平均，参数少，更抗位置变化。

小图可以用 Flatten；通道多、位置不重要时，GAP 更常见。

### 输入要归一化

合成图像是 0/1，IMU 幅度在 1 左右，所以脚本没有额外归一化。

真实数据要处理：

- 图像像素 `/255.0`，变成 0～1；
- IMU 每个轴减去均值、除以标准差，或者缩放到固定范围；
- 训练和推理必须用同一套归一化参数。

### 100% 不代表真实任务简单

这两个任务都是合成数据，类别差异非常明显。真实摄像头图像有光照、背景、遮挡；真实 IMU 有佩戴位置、个体差异。不要拿这里的 100% 去估计实际项目效果。

### `sparse` 和 one-hot

```text
标签是 0/1/2          -> sparse_categorical_crossentropy
标签是 [1,0,0] 这种   -> categorical_crossentropy
```

---

# 六、顺手改两下

1. 卷积核数量改成 4、8：参数量和准确率怎么变；
2. `padding="valid"`：看 shape 从 16×16 变成 14×14，再池化是 7×7；
3. 把图像模型的 `Flatten` 换成 `GlobalAveragePooling2D`：参数量会降很多，准确率还能保持吗；
4. 给图像加第三类（斜条、圆点）：最后一层改成 `Dense(3, softmax)`，loss 改成 `sparse_categorical_crossentropy`；
5. IMU 卷积核改成 5 和 3：感受野变小，还能分辨三个轴吗；
6. IMU 噪声从 0.1 加到 0.3：准确率下降，看混淆矩阵哪一类最先错。

## 到这儿你应该能

- 说清 `Conv2D` / `Conv1D` 的输入输出 shape；
- 自己算卷积层参数量；
- 说清 `padding="same"`、池化、`Flatten`、`GAP` 的作用；
- 搭一个二分类 CNN 和一个三分类 1D CNN；
- 说清 sigmoid + binary crossentropy 和 softmax + sparse categorical crossentropy 的搭配；
- 用混淆矩阵看每一类的错误；
- 解释为什么 CNN 的参数量不随图像尺寸爆炸。

下一步把模型导出成 int8 TFLite：先转 `.tflite`，在 PC 上用 `tf.lite.Interpreter` 验证，再考虑上板。
