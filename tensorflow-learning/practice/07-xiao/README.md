# 上 XIAO：TFLM + esp-nn 跑 int8 CNN

06 已经有：

```text
practice/06-tflite/image_cnn_int8.tflite
```

这一节把它放进 XIAO ESP32S3 的固件，用 TFLM 加载、esp-nn 执行 int8 推理。

先说明范围：这个固件**不接摄像头**，而是在固件内部生成两张 16×16 测试图，验证“模型进固件 → 算子注册 → 量化输入 → 推理 → 反量化输出”这条链路。摄像头接入和真实图像预处理放到 08。

## 工程位置

本机：

```text
/home/v/esp/xiao_esp32s3/tflite_cnn_hello/
```

学习项目里也放了一份精简源码：

```text
practice/07-xiao/firmware/
```

注意：`firmware/` 不包含 `managed_components/`、`sdkconfig` 和 `build/`。第一次构建时，IDF 会从 Espressif 组件仓库拉取 `esp-tflite-micro` 和 `esp-nn`；如果网络连不上 `components-file.espressif.com`，可以直接用本机已经下好依赖的工程，或者把依赖复制过来：

```bash
cp -a /home/v/esp/xiao_esp32s3/tflite_cnn_hello/managed_components practice/07-xiao/firmware/
cp /home/v/esp/xiao_esp32s3/tflite_cnn_hello/sdkconfig practice/07-xiao/firmware/
cp /home/v/esp/xiao_esp32s3/tflite_cnn_hello/dependencies.lock practice/07-xiao/firmware/
```

## 整条链路

```text
image_cnn_int8.tflite
  │ xxd -i
  ▼
main/model.cc
  g_model[] / g_model_len
  │ 编译进固件
  ▼
TFLM MicroInterpreter
  │ 注册 CONV_2D / MAX_POOL_2D / RESHAPE / FULLY_CONNECTED / LOGISTIC
  │ tensor_arena = 32 KB
  ▼
esp-nn int8 算子
  │
  ▼
串口输出 p(vertical)
```

## 模型怎么变成 C 数组

核心命令：

```bash
xxd -i image_cnn_int8.tflite > model.cc
```

`xxd -i` 会把二进制文件变成：

```c
unsigned char image_cnn_int8_tflite[] = { ... };
unsigned int image_cnn_int8_tflite_len = 5560;
```

但 TFLM 工程里需要的是：

```c
alignas(8) const unsigned char g_model[] = { ... };
const int g_model_len = 5560;
```

所以生成后要改符号名，并加上：

```c
#include "model.h"
```

`alignas(8)` 是为了让 FlatBuffers 按 8 字节对齐读取模型；不要漏掉。

这个工程里已经有生成好的 `main/model.cc`：

```text
g_model      5560 字节
g_model_len  5560
```

## 固件做了什么

### 1. 注册算子

模型里用到 5 个算子，resolver 里必须都注册：

```cpp
static tflite::MicroMutableOpResolver<5> resolver;
resolver.AddConv2D();
resolver.AddMaxPool2D();
resolver.AddReshape();
resolver.AddFullyConnected();
resolver.AddLogistic();
```

少注册一个，运行时会报：

```text
Didn't find op for builtin opcode ...
```

### 2. 分配 tensor arena

```cpp
constexpr int kTensorArenaSize = 32 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
```

`tensor_arena` 是 TFLM 给输入、输出、中间激活分配内存的地方。这个 CNN 很小，32 KB 足够；如果 `AllocateTensors()` 失败，先加大。

### 3. 输入量化

模型输入是 int8，不是 float32：

```text
shape = (1, 16, 16, 1)
scale = 0.003921568859368563
zero_point = -128
```

固件生成一张 16×16 的 0/1 图，写进输入张量前要量化：

```cpp
int32_t q = round(pixel / input->params.scale) + input->params.zero_point;
input->data.int8[y * 16 + x] = (int8_t)q;
```

### 4. 推理和输出反量化

```cpp
interpreter->Invoke();

float p = (output->data.int8[0] - output->params.zero_point)
        * output->params.scale;
```

输出是“竖条”的概率：

```text
p >= 0.5  -> vertical
p < 0.5   -> horizontal
```

## 构建

先加载 ESP-IDF：

```bash
. /home/v/esp/esp-idf/export.sh
```

然后进入工程。本机工程：

```bash
cd /home/v/esp/xiao_esp32s3/tflite_cnn_hello
```

如果看的是学习项目里的副本，从 `tensorflow-learning` 目录进：

```bash
cd practice/07-xiao/firmware
```

如果目录里还没有 `sdkconfig`，先设置目标芯片：

```bash
idf.py set-target esp32s3
```

然后构建：

```bash
idf.py build
```

注意：要进入 `firmware/` 再执行 `idf.py`。在 `07-xiao/` 这一层没有 `CMakeLists.txt`，会报 `CMakeLists.txt not found`。

本机构建结果：

```text
tflite_cnn_hello.bin  0x3df80 字节（约 253 KB）
app 分区剩余           87%
g_model                5560 字节
tensor_arena           0x8000（32 KB）
```

## 烧录和串口

板子连接后：

```bash
idf.py -p /dev/ttyACM0 flash monitor
```

如果提示端口被占用：

```text
Could not exclusively lock port /dev/ttyACM0
```

先关掉其他串口工具（包括另一个 `idf.py monitor`）：

```bash
fuser -v /dev/ttyACM0
```

如果 `/dev/ttyACM0` 不存在，先确认 USB 线是数据线、板子进了下载模式；VMware 下如果 `idf.py flash` 找不到端口，可以用 esptool 加 `--before=no_reset` 作为备选。

预期串口输出大致是：

```text
input type=9 shape=[1,16,16,1]
input scale=0.00392157 zero_point=-128
output type=9 shape=[1,1]
output scale=0.00390625 zero_point=-128
tensor arena used: ... bytes
model bytes: 5560

truth=horizontal p(vertical)=0.0000 pred=horizontal
truth=vertical   p(vertical)=0.9961 pred=vertical
```

PC 上 int8 验证的概率是 `0.0000` 和 `0.9961` 附近；板上会接近这个值，具体以串口为准。

这次只完成了构建；烧录需要板子连接，步骤见上。

## 为什么用 esp-nn

这个模型是 int8：

- `CONV_2D` 和 `FULLY_CONNECTED` 可以走 esp-nn 的 int8 优化 kernel；
- 不需要在板端做 float32 卷积；
- 推理速度和内存占用都比 float32 版本更适合 S3。

如果模型还是 float32，TFLM 会走通用参考实现，esp-nn 的 int8 加速就用不上。这也是 06 要转 int8 的原因之一。

## 常见问题

### `Didn't find op for builtin opcode`

resolver 少注册了算子。看 06 的算子列表：

```text
CONV_2D
MAX_POOL_2D
RESHAPE
FULLY_CONNECTED
LOGISTIC
```

### `AllocateTensors() failed`

tensor arena 太小。先加到 48 KB 或 64 KB 试。

### 输入 shape 或 dtype 不对

这个模型输入是 `(1,16,16,1)`、int8。不要把 256 个 float 直接写进 `data.int8`。

### 输出直接当概率用

int8 输出是 `-128~127`，必须：

```python
p = (q - zero_point) * scale
```

### 模型数组符号名不对

`model.h` 声明的是：

```c
extern const unsigned char g_model[];
extern const int g_model_len;
```

`xxd -i` 默认生成的 `image_cnn_int8_tflite` 要改成 `g_model`。

## 到这儿你应该能

- 解释 `.tflite` 怎么变成 `model.cc`；
- 说清 `g_model` / `g_model_len` / `alignas(8)`；
- 给 TFLM 注册模型需要的算子；
- 设置 `tensor_arena`；
- 在板端量化输入、反量化输出；
- 用 esp-nn 跑 int8 卷积和全连接。

下一步 08：把摄像头接进来，做真正的图像预处理和 person/no-person 二分类。
