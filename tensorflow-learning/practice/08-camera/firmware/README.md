# tflite_person firmware

08 的摄像头端到端工程源码。

```text
OV2640 -> 96×96 灰度 -> int8 -> TFLM + esp-nn -> person / no-person
```

## 构建

```bash
. /home/v/esp/esp-idf/export.sh
cd practice/08-camera/firmware
idf.py set-target esp32s3
idf.py build
```

如果网络连不上 Espressif 组件仓库，先从本机已经构建成功的工程复制依赖：

```bash
cp -a /home/v/esp/xiao_esp32s3/tflite_person/managed_components .
cp /home/v/esp/xiao_esp32s3/tflite_person/sdkconfig .
cp /home/v/esp/xiao_esp32s3/tflite_person/dependencies.lock .
```

## 烧录

```bash
idf.py -p /dev/ttyACM0 flash monitor
```

## 关键配置

```text
摄像头：FRAMESIZE_96X96，PIXFORMAT_GRAYSCALE
模型：  300,568 字节，int8
输入：  96×96×1
输出：  2 类，0=no-person，1=person
算子：  AveragePool2D / Conv2D / DepthwiseConv2D / Reshape / Softmax
arena： 100 KB + 60 KB esp-nn scratch
```

串口重点看：

```text
PERF capture=...ms invoke=...ms loop=...fps
INFO arena=.../...
person score:...%, no person score ...%
```
