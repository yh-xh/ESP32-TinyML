# 从 `.keras` 到 int8 `.tflite`

前面训练出来的模型是 Keras 格式，PC 上能跑，但 ESP32 上用的是 TFLite Micro。这一节把 05 的 CNN 转成 TFLite，并对比 float32 和 int8 两个版本。

代码里每个 API 都有注释；这份文档讲的是：为什么要量化、representative dataset 在干什么、int8 的输入输出怎么读。

## 先跑

```bash
conda activate tf-esp
cd practice/06-tflite
python convert_int8.py
```

这个脚本依赖 05 训练出的模型：

```text
practice/05-cnn/image_cnn.keras
```

如果找不到，先运行：

```bash
cd practice/05-cnn
python image_cnn.py
```

脚本会生成：

```text
image_cnn_float32.tflite
image_cnn_int8.tflite
```

这两个文件是生成的，不需要提交到 Git。

## 转换链路

```text
image_cnn.keras
  │ TFLiteConverter.from_keras_model
  ▼
float32 .tflite          不量化，输入输出 float32
  │ 加 representative_dataset
  │ 开启 Optimize.DEFAULT
  │ 输入输出设为 int8
  ▼
int8 .tflite             权重和激活量化成 int8
```

float32 版本用来做基准；int8 版本才是后面要放到板子上的。

## 为什么需要 representative dataset

int8 只有 256 个取值（-128～127），而 float32 的范围很大。转换器必须知道：

```text
这一层的输入大概在什么范围？
输出大概在什么范围？
```

它才能决定：

```text
scale：每个 int8 代表多大的 float 间隔
zero_point：float 的 0 对应哪个 int8
```

`representative_dataset` 就是拿一批有代表性的输入，让模型跑一遍，统计每层的 min/max。它不需要标签，只需要输入样本。

脚本里给了 100 张和训练集同分布的 16×16 横条/竖条图。真实项目里，校准数据要和实际输入同分布：摄像头图像、IMU 数据、归一化方式都要一致。

## 量化公式

```text
量化：   q = round(x / scale) + zero_point
反量化： x = (q - zero_point) × scale
```

这个模型实际得到的参数：

```text
input:
  dtype int8
  shape (1, 16, 16, 1)
  scale = 0.003921568859368563  (约 1/255)
  zero_point = -128

output:
  dtype int8
  shape (1, 1)
  scale = 0.00390625  (1/256)
  zero_point = -128
```

所以输入 0 会映射到 `-128`，输入 1 会映射到 `127`。

输出是 sigmoid 后的概率，但模型给你的仍然是 int8：

```text
q = -128  -> 概率约 0
q = 127   -> 概率约 0.996
```

要得到概率，必须反量化：

```python
prob = (q - output_zero_point) * output_scale
```

不要直接把 int8 的 `-128/127` 当概率用。

## 实际结果

```text
float32 tflite: 8988 bytes
int8 tflite:    5560 bytes
```

这个模型太小，文件头、元数据和张量描述占了不少空间，所以 int8 不是严格 1/4。模型越大，int8 的体积优势越明显。

PC 上 200 张测试图：

```text
accuracy:
  keras:           1.0
  float32 tflite:  1.0
  int8 tflite:     1.0

probability difference:
  max |keras - float32|: 2.1e-09
  max |float32 - int8|:  0.00390
  mean |float32 - int8|: 0.00194
```

这个任务很简单，int8 没有掉点。真实模型上，量化误差取决于：

- 校准数据是否覆盖真实输入分布；
- 模型对权重/激活误差是否敏感；
- 是否有离群值；
- 量化方案（仅权重、全 int8、混合精度）。

## 模型里有哪些算子

用 `tf.lite.Interpreter` 看 int8 模型，主要算子是：

```text
CONV_2D
MAX_POOL_2D
CONV_2D
MAX_POOL_2D
RESHAPE
FULLY_CONNECTED
LOGISTIC
```

对应：

```text
Conv2D          -> CONV_2D
MaxPooling2D    -> MAX_POOL_2D
Flatten         -> RESHAPE
Dense           -> FULLY_CONNECTED
sigmoid         -> LOGISTIC
```

这些都在 TFLM 的支持范围内；int8 的 `CONV_2D` 和 `FULLY_CONNECTED` 也是 esp-nn 主要加速的算子。

## 用 Netron 看模型

1. 打开 <https://netron.app/>；
2. 把 `image_cnn_int8.tflite` 拖进去；
3. 点输入/输出节点，看 dtype、shape 和 quantization；
4. 看中间节点，确认是 `CONV_2D` / `MAX_POOL_2D` / `FULLY_CONNECTED`。

Netron 不运行模型，只显示图结构。它能帮你确认：

- 输入是不是 int8；
- 输出是不是 int8；
- 有没有出现 TFLM 不支持的算子；
- 量化参数是不是写进了模型。

## 转换日志里的警告

转换时可能看到类似：

```text
Non-Converted Ops: ...
% non-converted = ...
ARITH ops
```

先不要慌。判断标准不是日志里有没有警告，而是：

1. `converter.convert()` 是否成功返回 bytes；
2. `.tflite` 文件是否生成；
3. `tf.lite.Interpreter` 是否能 `allocate_tensors()` 并 `invoke()`；
4. 输出和 float32 版本是否接近。

这个脚本三项都通过了，所以模型可用。

## 常见坑

### 忘记 representative dataset

不提供校准数据，全 int8 转换会报错，或者量化参数完全不对。

### 输入 dtype 不匹配

int8 模型的输入是 `np.int8`，不是 `float32`。调用 `set_tensor` 前要按 input 的 `scale/zero_point` 量化。

### 输出忘了反量化

int8 输出是 `-128~127`，不是概率。必须：

```python
prob = (q - zero_point) * scale
```

### 校准数据和真实数据分布不一致

比如用纯随机噪声校准，但实际输入是摄像头图像，量化范围会偏。校准集要尽量接近真实输入。

### 只量化了权重

`converter.optimizations = [Optimize.DEFAULT]` 默认是训练后量化；如果不设置输入输出为 int8，模型输入的仍是 float32。端侧要省掉 float 转换，通常把 `inference_input_type` / `inference_output_type` 都设为 int8。

### 以为 int8 一定掉点

小模型、简单任务可能几乎不掉；复杂模型可能掉几个点。必须用测试集对比，不能只看文件大小。

## 顺手改两下

1. 把 representative dataset 从 100 张改成 10 张，看量化误差是否变大；
2. 把校准图换成纯随机噪声，看输出概率和准确率怎么变；
3. 把 05 的 `imu_cnn.keras` 也转一遍，比较 float32/int8 大小；
4. 在 Netron 里对比 float32 和 int8 模型的输入/输出节点；
5. 打印更多测试样本的 float32 和 int8 概率，找误差最大的那张图。

## 到这儿你应该能

- 说清 `.keras`、float32 `.tflite`、int8 `.tflite` 的区别；
- 说清 representative dataset 的作用；
- 用 `TFLiteConverter` 转全 int8 模型；
- 用 `tf.lite.Interpreter` 检查输入输出 shape、dtype、scale、zero_point；
- 自己量化输入、反量化输出；
- 用 Netron 看 TFLite 的算子；
- 解释为什么 int8 文件不一定刚好是 float32 的 1/4。

下一步把 int8 `.tflite` 转成 C 数组，放进 XIAO 的 TFLM 工程，用 esp-nn 跑起来。
