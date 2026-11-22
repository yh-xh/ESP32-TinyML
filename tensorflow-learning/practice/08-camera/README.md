# 08：摄像头端到端，person / no-person

前面 07 用的是固件里硬编码的 16×16 测试图，只验证了 TFLM 链路。这一节接真实摄像头，走一遍：

```text
OV2640 取帧
  → 96×96 灰度
  → uint8 转 int8
  → TFLM + esp-nn 推理
  → 反量化 person / no-person 分数
  → 串口输出 + 耗时统计
```

这一节用现成的 `tflite_person` 参考工程，先把真实链路跑通、把数字量出来；把它变成“你自己的项目”，关键在最后的重训练和换模型。

## 参考工程

本机：

```text
/home/v/esp/xiao_esp32s3/tflite_person/
```

它已经包含：

```text
OV2640 初始化
96×96 灰度摄像头配置
int8 person detection 模型（300,568 字节）
TFLM 解释器 + esp-nn
串口耗时和 arena 统计
```

## 先跑起来

```bash
source /home/v/esp/esp-idf/export.sh
cd /home/v/esp/xiao_esp32s3/tflite_person
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

如果端口被占用，先关掉其他 `idf.py monitor`：

```bash
fuser -v /dev/ttyACM0
```

退出 monitor：`Ctrl+]`。

## 代码地图

### 摄像头初始化

`main/app_camera_esp.c`：

```text
config.frame_size   = FRAMESIZE_96X96
config.pixel_format = PIXFORMAT_GRAYSCALE
config.fb_count     = 2
config.fb_location  = CAMERA_FB_IN_PSRAM
```

摄像头直接输出 96×96 灰度，和模型输入一致，所以不需要在固件里做缩放。

### 取帧和量化

`main/image_provider.cc` 的 `GetImage()`：

```cpp
camera_fb_t* fb = esp_camera_fb_get();
for (int i = 0; i < image_width * image_height; i++) {
    image_data[i] = ((uint8_t *) fb->buf)[i] ^ 0x80;
}
esp_camera_fb_return(fb);
```

`fb->buf` 是 uint8 灰度（0～255），模型输入是 int8（-128～127）。

```text
uint8 0   -> int8 -128
uint8 128 -> int8 0
uint8 255 -> int8 127
```

`^ 0x80` 是快速把最高位取反，效果等价于 `value - 128`。它成立的前提是模型输入的 `scale/zero_point` 和这个映射匹配；换模型时要重新确认。

### 模型和算子

`main/model_settings.h`：

```cpp
constexpr int kNumCols = 96;
constexpr int kNumRows = 96;
constexpr int kNumChannels = 1;
constexpr int kCategoryCount = 2;
constexpr int kPersonIndex = 1;
constexpr int kNotAPersonIndex = 0;
```

`main/main_functions.cc` 注册 5 个算子：

```text
AveragePool2D
Conv2D
DepthwiseConv2D
Reshape
Softmax
```

参考模型大小：

```text
300,568 字节
```

### arena

```cpp
#if CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN
constexpr int scratchBufSize = 60 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif

constexpr int kTensorArenaSize = 100 * 1024 + scratchBufSize;
```

- 100 KB 给 TFLM 的输入、输出、中间激活；
- 60 KB 是 esp-nn 卷积/深度卷积的 scratch buffer；
- 优先放内部 RAM，失败再回退到 PSRAM。

换更大的模型时，先看 `AllocateTensors()` 是否成功，再看 `arena_used_bytes()`。

### 输出

```cpp
int8_t person_score = output->data.int8[kPersonIndex];
float p = (person_score - output->params.zero_point) * output->params.scale;
```

不要直接把 int8 当概率。`RespondToDetection()` 里用 60% 作为阈值：

```text
person_score >= 60% -> person
否则                -> no-person
```

阈值不是模型的一部分，是你按误检/漏检需求调的。

## 串口能看到的数字

参考工程会打印：

```text
input: ... scale ... zero_point ...
output: ... scale ... zero_point ...
arena: requested ... bytes, used ... bytes
INFO arena=.../... input=... output=... out_scale=... out_zp=...
PERF capture=...ms invoke=...ms loop=...fps
person score:...%, no person score ...%
```

重点看三个：

```text
capture  取帧 + 灰度/量化耗时
invoke   模型推理耗时
loop     整体循环帧率
```

这三个数比“模型多少 KB”更能说明端侧体验。记录时写清楚：

- 分辨率、灰度/彩色；
- 是否启用 esp-nn；
- arena 用了多少；
- 测试场景（室内/室外、明暗、距离）；
- 误检和漏检的例子。

## 把它变成你自己的项目

参考工程用的是现成模型。要变成自己的项目，走这条线：

```text
采集数据 -> 标注 -> 训练 -> int8 量化 -> 检查模型 -> 替换模型 -> 板端复测
```

### 1. 采数据

用摄像头采，不要只用网图。目标是 person / no-person 二分类：

- person：不同距离、姿态、光照、背景；
- no-person：空房间、椅子、宠物、海报、屏幕；
- 两个类别数量尽量平衡；
- 训练、验证、测试按场景分开，不要同一段视频随机切；
- 测试集只在最后用。

如果模型输入是 96×96 灰度，采集和预处理就按 96×96 灰度做，别训练用彩色、上板用灰度。

### 2. 训练

可以直接复用 05 的 CNN 结构，把输入改成：

```text
(96, 96, 1)
```

最后一层：

```python
Dense(1, activation="sigmoid")
loss="binary_crossentropy"
```

或者二分类输出 2 类 softmax。关键是训练和板端的标签顺序一致：

```text
0 = no-person
1 = person
```

### 3. 量化

按 06 的流程：

```text
代表数据集 -> TFLiteConverter -> 全 int8
input/output int8
```

代表数据集要用真实摄像头图，不要用随机噪声。

### 4. 检查模型

先用 `check_model.py` 看：

```bash
conda activate tf-esp
cd practice/08-camera
python check_model.py path/to/your_model.tflite
```

确认：

- 输入 shape 是 `[1,96,96,1]`；
- 输入是 int8；
- 输出类别数和 `model_settings.h` 一致；
- 算子都在 TFLM 支持范围内。

### 5. 替换模型

参考工程用的是：

```text
main/person_detect_model_data.cc
  g_person_detect_model_data[]
  g_person_detect_model_data_len
```

用 `xxd -i` 生成新数组，改符号名，替换即可。然后检查：

- `model_settings.h` 的宽高、通道、类别数、person index；
- `main_functions.cc` 的 resolver 是否覆盖新模型的算子；
- arena 是否够；
- 输入量化方式和 `GetImage()` 的 uint8→int8 转换是否匹配；
- 输出反量化和阈值。

### 6. 板端复测

不要只看 PC 准确率。上板记录：

```text
准确率 / 误检 / 漏检
capture / invoke / loop
arena used
不同光照和距离
```

## 常见坑

### 训练和板端预处理不一致

训练用 RGB、板端用灰度；训练归一化到 0～1、板端直接 int8；训练 resize 方式不同。这些都会让模型“看起来训练很好，上板很差”。

### 标签顺序反了

训练时 `0=no-person, 1=person`，板端 `kPersonIndex` 却写成 0。输出分数会反。

### 模型输入 shape 不匹配

参考工程固定 96×96×1。换模型后先看 `check_model.py` 的输入 shape。

### 算子没注册

TFLM 只链接你注册的算子。新模型多了 `MEAN`、`ADD`、`SOFTMAX` 等，就要在 resolver 里加。

### arena 太小

`AllocateTensors() failed` 时先加 arena；但 arena 不是越大越好，内部 RAM 紧张时会影响摄像头和 Wi-Fi。

### 摄像头取帧失败

检查摄像头排线、BSP 引脚、`fb_count` 和 PSRAM 配置。XIAO Sense 的 SD 卡和 LED 引脚有冲突，接线时注意。

## 顺手改两下

1. 把 person 阈值从 60% 改成 80%，看误检和漏检怎么变；
2. 记录 1 分钟内的 `PERF` 平均值，比较 capture 和 invoke 谁更耗时；
3. 用静态图片 CLI 模式跑几张图，和摄像头实时结果对比；
4. 把模型换成 06 的 16×16 玩具 CNN，观察输入 shape 和 `model_settings.h` 需要改哪些地方（只做链路验证，不要期待识别行人）；
5. 采集 50 张自己的 person/no-person 图，先训练一个能过拟合的小模型，再谈泛化。

## 到这儿你应该能

- 说清摄像头 → 灰度 → int8 → TFLM → 输出分数的完整链路；
- 解释 `^ 0x80`、输入 scale/zero_point、输出反量化；
- 看懂 `PERF` 和 `INFO` 日志；
- 把新模型换进参考工程，并检查 shape、算子、arena、标签顺序；
- 说清“参考模型能跑”和“自己的数据训练”之间还差什么。

下一步就是真正用你自己的数据训练、量化、替换模型，并把误检/漏检和耗时记录成一张表。
