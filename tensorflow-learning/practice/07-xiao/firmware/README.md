# tflite_cnn_hello

XIAO ESP32S3 上运行 `image_cnn_int8.tflite` 的最小 TFLM + esp-nn 工程。

模型来自 `../../06-tflite/image_cnn_int8.tflite`，已经用 `xxd -i`
转成 `main/model.cc` 里的 `g_model` 数组。

## 构建

先加载 ESP-IDF 环境：

```bash
. /home/v/esp/esp-idf/export.sh
cd practice/07-xiao/firmware
idf.py build
```

## 烧录

确认串口后：

```bash
idf.py -p /dev/ttyACM0 flash monitor
```

退出 monitor：`Ctrl+]`。

## 固件在做什么

固件不接摄像头，而是在内部生成两张 16×16 测试图：

```text
横条：中间两行 = 1，其余 = 0
竖条：中间两列 = 1，其余 = 0
```

每 500ms 交替跑一次推理，串口打印：

```text
truth=horizontal p(vertical)=... pred=...
truth=vertical   p(vertical)=... pred=...
```

关键文件：

```text
main/model.cc            g_model / g_model_len
main/main_functions.cc   TFLM 初始化、量化输入、反量化输出
main/CMakeLists.txt      只编译 main.cc、main_functions.cc、model.cc
```

模型需要 5 个算子：

```text
CONV_2D
MAX_POOL_2D
RESHAPE
FULLY_CONNECTED
LOGISTIC
```

`tensor_arena` 设为 32 KB；如果 `AllocateTensors()` 失败，先加大 arena。
