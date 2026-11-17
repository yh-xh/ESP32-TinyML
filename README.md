# XIAO ESP32-S3 / esp-nn + TFLite Micro 学习笔记

> **适用环境**：ESP-IDF v5.5.3 · esp-tflite-micro 1.4.3 · esp-nn 1.5.0 · esp32-camera 2.1.9 · Seeed Studio XIAO ESP32S3 Sense（OV2640）
> **记录时间**：2026-11
> 内容以实测和源码为准，版本更新后部分 API 可能变化。
> 配套的 esp-dl 笔记见 [esp32-p4 分支](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-p4)，手势识别笔记见 [esp32-c5 分支](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-c5)。

> **先对齐概念**：本文实战部署的是官方 `person_detection` 二分类模型，类别是 **person / no-person（有人 / 无人）**。它只回答“画面里有没有人”，不输出人脸框，也不回答“是谁”。
> 如果你的目标是 **人脸 / 非人脸** 二分类，替换模型和标签即可，部署流程完全一致，见第 9 章；如果你的目标是 **识别是谁**，那属于人脸识别（embedding + 相似度），不是二分类，见 9.7。

---

## 目录

- [1. XIAO ESP32-S3 与 esp-nn 是什么](#1-xiao-esp32-s3-与-esp-nn-是什么)
- [2. TFLM 特有内容（公共基础见主分支）](#2-tflm-特有内容公共基础见主分支)
- [3. 从模型到芯片的完整流程](#3-从模型到芯片的完整流程)
- [4. 实战：XIAO ESP32S3 Sense 二分类人形检测](#4-实战xiao-esp32s3-sense-二分类人形检测)
- [5. TFLite Micro 运行时与 esp-nn](#5-tflite-micro-运行时与-esp-nn)
- [6. XIAO ESP32-S3 平台配置](#6-xiao-esp32-s3-平台配置)
- [7. 性能与量化](#7-性能与量化)
- [8. 实战踩坑（重点）](#8-实战踩坑重点)
- [9. 换成自己的二分类模型（人脸 / 非人脸）](#9-换成自己的二分类模型人脸--非人脸)
- [10. 学习路线与速查命令](#10-学习路线与速查命令)
- [11. 参考](#11-参考)

---

## 1. XIAO ESP32-S3 与 esp-nn 是什么

### 1.1 开发板

Seeed Studio XIAO ESP32S3 **Sense** 是“XIAO ESP32S3 主板 + Sense 扩展板”的组合。本文用到的是扩展板上的 OV2640 摄像头。

| 项 | 值 |
|---|---|
| SoC | ESP32-S3R8，双核 Xtensa LX7，最高 240 MHz |
| 内部 SRAM | 512 KB |
| PSRAM | 8 MB，八线（Octal） |
| Flash | 8 MB |
| 摄像头 | OV2640，200 万像素，DVP 8 位并口 |
| 麦克风 | PDM 数字麦克风（CLK=GPIO42，DATA=GPIO41） |
| 存储 | microSD 卡槽（SCK=GPIO7，MISO=GPIO8，MOSI=GPIO9，CS=GPIO21） |
| 用户 LED | GPIO21（和 SD 的 CS 共用） |
| 充电指示灯 | CHG-LED，硬件控制，不是 GPIO |
| 无线 | 2.4 GHz Wi-Fi + BLE 5.0 |
| 调试/供电 | USB-C，原生 USB Serial/JTAG（`/dev/ttyACM0`） |

> **注意**：GPIO21 同时是 SD 卡 CS 和用户 LED。插卡使用时 LED 会受影响；本文的 person_detection 不用 SD，所以不涉及。板子上的另一个 LED 是充电指示灯，不能当普通 GPIO 用。

### 1.2 esp-nn 是什么

**esp-nn 是 Espressif 的 int8 神经网络算子库**，不是推理框架，也不是模型格式。

它只负责把卷积、深度卷积、池化、全连接、softmax 等算子写成芯片上的高性能实现。模型解析、张量分配、算子调度由上层运行时负责——在 ESP-IDF 里主要是 **TensorFlow Lite Micro（TFLM）**。

几个关键事实：

- esp-nn 主要服务于 TFLite Micro，官方 README 的 Supported platforms 写的就是 TFLite Micro。
- 内核以 int8（s8）为主，部分 s16；不是通用 float 推理库。
- 按芯片分级优化：**ESP32-S3 和 ESP32-P4 有汇编/SIMD 专用路径**；ESP32-C5 等更多走通用优化 C 或 ANSI C。
- 不依赖 esp-dl；esp-dl 自带算子实现，是另一条并列路线。

### 1.3 esp-nn 在 TinyML / TFLite 生态中的位置

**TinyML = Tiny Machine Learning（微型机器学习）**，指在 MCU 和超低功耗设备上直接跑推理。它是一个领域，不是具体框架。ESP32 上常见两条路线：

```
TinyML（领域）
├── 路线 A：TFLite Micro + esp-nn    ← 本文走的路线
└── 路线 B：esp-dl + esp-ppq         ← P4 笔记里的路线
```

| 名称 | 是什么 | 不是什么 |
|---|---|---|
| **TinyML** | 领域：资源受限 MCU 上的本地推理 | 不是某个具体框架 |
| **TFLite Micro (TFLM)** | 运行时 + 模型格式（`.tflite`，FlatBuffers） | 不是算子库，也不是 TinyML 的同义词 |
| **esp-nn** | Espressif 的 int8 算子库，TFLM 在 ESP32 上的加速后端 | 不是运行时，不解析模型 |
| **esp-dl** | Espressif 的另一套运行时 + `.espdl` 格式 + 模型库 | 不依赖 esp-nn，走自己的 SIMD/ISA |

### 1.4 两条路线对比

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

### 1.5 本文到底部署了什么

本文的工程基于 `esp-tflite-micro` 官方 `person_detection` 示例，部署的是它的预训练 int8 模型：

| 项 | 值 |
|---|---|
| 任务 | 二分类：person / no-person |
| 输入 | 96 × 96 × 1，灰度，int8 |
| 输出 | 2 个 int8 分数，反量化后是两类概率 |
| 模型大小 | 300,568 字节（约 293.5 KiB） |
| 模型来源 | TensorFlow / TFLite Micro 的 person detection 示例模型 |
| 输出含义 | 整张图里“有没有人”的分数，不是人脸框，也不是身份 |

这个模型**不是**人脸识别模型：

- 它不会告诉你“这是谁”；
- 它不会给人脸画框；
- 它把整张 96×96 图当成一个样本，输出两类概率。

如果你要的是“人脸 / 非人脸”的二分类，模型结构可以完全一样，只是训练数据和标签不同；如果你要“识别某个人”，请看 9.7。

---

## 2. TFLM 特有内容（公共基础见主分支）

> 模型组成、推理 vs 训练、张量/布局、常见文件格式、量化概念、FlatBuffers、常识校正
> 是两份笔记的公共基础，已抽离到主分支统一维护：
> **[model-deployment.md · 模型与部署基础](https://github.com/yh-xh/ESP32-TinyML/blob/main/model-deployment.md)**
>
> 本章只保留 TFLM / 本项目特有的内容。

### 2.1 本项目的张量与内存布局

- 输入张量形状是 `[1, 96, 96, 1]`：batch=1，96×96 灰度，单通道。
- 输入类型是 **int8**，共 `96 × 96 = 9,216` 字节。
- 摄像头输出的是 uint8 灰度（0–255），`image_provider.cc` 里用：

  ```c
  image_data[i] = ((uint8_t *) fb->buf)[i] ^ 0x80;
  ```

  把 0–255 映射到 int8 的 -128–127。这只是一个位映射；真实数值范围由输入张量的 `scale / zero_point` 决定，**自定义模型不要无脑照抄**，要按 `input->params` 计算。
- 输出张量形状是 `[1, 2]`，int8，共 2 字节；类别顺序由训练决定：

  | index | 标签 | 含义 |
  |---:|---|---|
  | 0 | `notperson` | 无人 |
  | 1 | `person` | 有人 |

  代码里对应 `kNotAPersonIndex = 0`、`kPersonIndex = 1`。**索引反了，结果就会反过来。**
- 实测输出量化参数：`scale = 0.00390625`（即 1/256），`zero_point = -128`。反量化公式：

  ```c
  float score = (raw - output->params.zero_point) * output->params.scale;
  ```

- 代码里有一处容易误会的地方：

  ```c
  int8_t person_score = output->data.uint8[kPersonIndex];
  ```

  `TfLiteTensor::data` 是 union，int8 张量的字节既可以通过 `data.int8` 访问，也可以通过 `data.uint8` 访问；这里用 `uint8` 取原始字节，再赋给 `int8_t`，位模式不变。自己写新代码时建议直接用 `data.int8`，可读性更好。

### 2.2 TFLite 量化的补充

- 本模型是**全整数 int8 量化**：权重、激活、输入、输出都是 int8；不是 float32 模型。
- 全整数量化在 PC 端需要 `representative_dataset`，否则只能做动态范围量化或保留 float。
- 输出是 softmax 两类，两个分数加起来约 100%；二分类可以直接取较大者，也可以用阈值（示例的显示路径用 60% 作为“有人”的门槛）。
- 量化后的分数分辨率是 `scale = 1/256`，所以串口打印的百分比会有 1% 左右的跳变，这是正常的。

### 2.3 一次推理在 TFLM 里的生命周期

```
1. 加载    tflite::GetModel(g_person_detect_model_data) 拿到 FlatBuffer 指针
2. 注册    MicroMutableOpResolver<5> 注册模型用到的 5 个算子
3. 分配    MicroInterpreter + tensor_arena，AllocateTensors()
4. 取图    esp_camera_fb_get() → 96×96 灰度 → ^0x80 → input->data.int8
5. 前向    interpreter->Invoke()
6. 取输出  output->data.int8[2] → 反量化 → person / no-person
```

没有动态内存分配，所有中间张量都在 `tensor_arena` 里复用；`batch` 只能是 1。

### 2.4 模型放哪里

本项目用最直接的方式：模型 C 数组编进固件，放在 flash 的 `.rodata`。

其他方式（`EMBED_FILES`、独立 flash 分区、SD/文件系统、PSRAM 缓冲区）在 9.6 节展开；核心不变：`tflite::GetModel()` 只接受一个**可寻址内存指针**，模型必须在推理期间保持有效且对齐。

---

## 3. 从模型到芯片的完整流程

本项目的全链路：

```
官方 person_detect.tflite（300,568 字节，int8）
  → xxd -i / convert_bytes_to_c_source()
  → person_detect_model_data.cc（约 1.9 MB 源码）
  → 编进固件 flash 的 .rodata
  → TFLM：GetModel → MicroInterpreter → AllocateTensors
  → OV2640 输出 96×96 灰度
  → ^0x80 转 int8 → input->data.int8
  → Invoke()（esp-nn int8 内核）
  → output[2] → 反量化 → person / no-person
```

关键约束：

| 约束 | 本项目的值 | 说明 |
|---|---|---|
| 输入尺寸 | 96 × 96 × 1 | 改尺寸必须换模型 |
| 输入类型 | int8 | 不是 uint8，也不是 float |
| batch | 1 | TFLM 只做单样本推理 |
| 输出类别 | 2 | 索引顺序必须和训练一致 |
| 模型位置 | flash `.rodata` | 300,568 字节，不占 RAM |
| 推理频率 | 每次采集后一次 | 实测约 49.6 ms/次 |

> 本文**不训练模型**，直接用官方预训练模型。要训练自己的“人脸 / 非人脸”二分类模型，按第 9 章走一遍：采数据 → 训练 → 转 int8 `.tflite` → 替换模型 → 改标签和输入尺寸。

---

## 4. 实战：XIAO ESP32S3 Sense 二分类人形检测

### 4.1 项目结构

工程基于 `esp-tflite-micro` 的 `person_detection` 示例，目录名 `tflite_person`。

| 文件 | 作用 |
|---|---|
| `main/main.cc` | `app_main`：创建 `tf_main` 任务（8 KB 栈），启动 CLI 和摄像头推理循环 |
| `main/main_functions.cc` | 模型加载、算子注册、arena 分配、`loop()` 推理、耗时打点 |
| `main/app_camera_esp.c` | `camera_config_t` 填写、`esp_camera_init()`、传感器翻转 |
| `main/app_camera_esp.h` | 摄像头引脚宏、96×96、灰度、XCLK 20 MHz |
| `main/image_provider.cc` | `GetImage()`：取帧、灰度转 int8、归还帧缓冲 |
| `main/detection_responder.cc` | 打印 person / no-person 分数；有显示屏时控制 LED |
| `main/model_settings.h/.cc` | 96×96×1、2 类、标签顺序 |
| `main/person_detect_model_data.cc` | 模型 C 数组（300,568 字节） |
| `main/Kconfig.projbuild` | BSP / 摄像头引脚选择（自定义引脚在这里配置） |
| `main/idf_component.yml` | `esp-tflite-micro`、`esp32-camera` 等依赖 |
| `partitions.csv` | 自定义 OTA 分区表 |
| `sdkconfig.defaults` | 工程默认配置（实际以 `sdkconfig` 为准） |
| `static_images/` | 10 张 96×96 原始图，CLI 模式下的 `detect_image` 用 |

### 4.2 硬件与引脚

**摄像头（OV2640，DVP）**：

| 信号 | GPIO | 说明 |
|---|---:|---|
| XCLK | 10 | 摄像头主时钟，20 MHz |
| SIOD（SCCB SDA） | 40 | 摄像头寄存器 I2C 数据 |
| SIOC（SCCB SCL） | 39 | 摄像头寄存器 I2C 时钟 |
| VSYNC | 38 | 场同步 |
| HREF | 47 | 行有效 |
| PCLK | 13 | 像素时钟 |
| D0（Y2） | 15 | 数据位 0 |
| D1（Y3） | 17 | 数据位 1 |
| D2（Y4） | 18 | 数据位 2 |
| D3（Y5） | 16 | 数据位 3 |
| D4（Y6） | 14 | 数据位 4 |
| D5（Y7） | 12 | 数据位 5 |
| D6（Y8） | 11 | 数据位 6 |
| D7（Y9） | 48 | 数据位 7 |
| PWDN | -1 | 未接/不受控 |
| RESET | -1 | 未接/软复位 |

**摄像头参数**：

| 项 | 值 |
|---|---|
| 像素格式 | `PIXFORMAT_GRAYSCALE`（无显示屏时） |
| 分辨率 | `FRAMESIZE_96X96` |
| XCLK | 20 MHz |
| 帧缓冲数量 | 2（`fb_count = 2`） |
| 帧缓冲位置 | PSRAM（`CAMERA_FB_IN_PSRAM`） |
| 垂直翻转 | `set_vflip(s, 1)` |

**其他外设**：

- PDM 麦克风：CLK=GPIO42，DATA=GPIO41；本 demo 不用。
- SD 卡：SCK=7，MISO=8，MOSI=9，CS=21；本 demo 不用。
- 用户 LED：GPIO21，和 SD CS 冲突；本 demo 不点灯。
- 充电指示灯：CHG-LED，硬件控制。

> **GPIO48 的坑**：ESP32-S3 的 GPIO48 是合法引脚，但官方 `person_detection` 示例的 `main/Kconfig.projbuild` 里，自定义摄像头引脚的 `range` 上限原本是 39（部分选项 33），在 menuconfig 里根本选不到 48。必须先把范围改成 48，见 4.3。

### 4.3 从官方示例移植到 XIAO

#### 第 1 步：创建工程并切目标

在 `esp-tflite-micro` 的 `person_detection` 示例基础上创建工程（或直接复制示例目录），然后：

```bash
. $IDF_PATH/export.sh
idf.py set-target esp32s3
```

#### 第 2 步：打开 8 MB Flash 和 Octal PSRAM

XIAO ESP32S3 Sense 是 8 MB Flash + 8 MB PSRAM，摄像头帧缓冲要放 PSRAM。关键配置：

| 配置 | 值 |
|---|---|
| `CONFIG_ESPTOOLPY_FLASHSIZE_8MB` | `y` |
| `CONFIG_ESPTOOLPY_FLASHMODE_QIO` | `y` |
| `CONFIG_ESPTOOLPY_FLASHFREQ_80M` | `y` |
| `CONFIG_SPIRAM` | `y` |
| `CONFIG_SPIRAM_MODE_OCT` | `y` |
| `CONFIG_SPIRAM_SPEED_80M` | `y` |
| `CONFIG_SPIRAM_USE_CAPS_ALLOC` | `y` |
| `CONFIG_ESP32S3_DEFAULT_CPU_FREQ_240` | `y` |
| `CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG` | `y` |
| `CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN` | `y` |
| `CONFIG_NN_OPTIMIZED` | `y` |
| `CONFIG_NN_OPTIMIZATIONS` | `1` |

启动日志里应该能看到：

```
I (186) octal_psram: vendor id    : 0x0d (AP)
I (186) octal_psram: density      : 0x03 (64 Mbit)
I (186) octal_psram: Readlatency  : 0x02 (10 cycles@Fixed)
```

`64 Mbit` 就是 8 MB。

#### 第 3 步：选择自定义摄像头引脚

`idf.py menuconfig`：

```
Application Configuration
  └── Camera Configuration
        ├── Select Camera Pinout → Custom Camera Pinout
        ├── Power Down pin = -1
        ├── Reset pin = -1
        ├── XCLK = 10
        ├── SIOD = 40
        ├── SIOC = 39
        ├── VSYNC = 38
        ├── HREF = 47
        ├── PCLK = 13
        ├── Y2 = 15
        ├── Y3 = 17
        ├── Y4 = 18
        ├── Y5 = 16
        ├── Y6 = 14
        ├── Y7 = 12
        ├── Y8 = 11
        └── Y9 = 48
```

如果 menuconfig 里 GPIO48 选不到，先改 `main/Kconfig.projbuild` 的 `range`：

```kconfig
config CAMERA_PIN_Y9
    depends on CAMERA_MODULE_CUSTOM
    int "Y9 pin"
    range 0 48        # 原示例可能是 0 39
    default 35
```

PWDN/RESET 的 `range -1 48` 同理。改完执行：

```bash
idf.py reconfigure
idf.py menuconfig
```

Kconfig 缓存不刷新时，改了 `range` 也可能看不到新值。

#### 第 4 步：固定 96×96 灰度

`main/app_camera_esp.h`：

```c
#if defined DISPLAY_SUPPORT
#define CAMERA_PIXEL_FORMAT PIXFORMAT_RGB565
#else
#define CAMERA_PIXEL_FORMAT PIXFORMAT_GRAYSCALE
#endif

#define CAMERA_FRAME_SIZE FRAMESIZE_96X96
#define XCLK_FREQ_HZ 20000000
```

本项目没有显示屏，`DISPLAY_SUPPORT` 未定义，所以走灰度路径。如果接了 BSP 显示屏，摄像机会输出 RGB565，再由 `scale_rgb565_to_gray()` 转灰度，这条路径更耗 CPU，也会占更多 PSRAM。

#### 第 5 步：摄像头初始化

`main/app_camera_esp.c` 的关键代码：

```c
camera_config_t config;
config.ledc_channel = LEDC_CHANNEL_0;
config.ledc_timer = LEDC_TIMER_0;
config.pin_d0 = CAMERA_PIN_D0;
/* ... d1 ~ d7、xclk、pclk、vsync、href、sccb ... */
config.pin_pwdn = CAMERA_PIN_PWDN;
config.pin_reset = CAMERA_PIN_RESET;
config.xclk_freq_hz = XCLK_FREQ_HZ;
config.jpeg_quality = 10;
config.fb_count = 2;
config.fb_location = CAMERA_FB_IN_PSRAM;
config.pixel_format = CAMERA_PIXEL_FORMAT;
config.frame_size = CAMERA_FRAME_SIZE;

esp_err_t err = esp_camera_init(&config);
if (err != ESP_OK) {
    ESP_LOGE(TAG, "Camera init failed with error 0x%x", err);
    return -1;
}
sensor_t *s = esp_camera_sensor_get();
s->set_vflip(s, 1);
```

#### 第 6 步：任务栈和主循环

`main/main.cc` 里 `tf_main` 的栈从示例的 4 KB 提高到 8 KB：

```c
xTaskCreate((TaskFunction_t)&tf_main, "tf_main", 8 * 1024, NULL, 8, NULL);
```

摄像头初始化、TFLM 解释器和 CLI 都会用栈，4 KB 容易溢出。`loop()` 在 `tf_main` 里跑，`vTaskDelay(1)` 既让出 CPU，也避免任务看门狗。

#### 第 7 步：编译、烧录、看日志

```bash
idf.py build
idf.py -p /dev/ttyACM0 flash
```

本项目实测构建结果：

```
person_detection.bin binary size 0xb94f0 bytes.
Smallest app partition is 0x1e0000 bytes.
0x126b10 bytes (61%) free.
```

烧录偏移不是常见的 `0x8000/0x10000`，而是：

| 镜像 | 偏移 |
|---|---:|
| bootloader | `0x0` |
| partition-table | `0xc000` |
| ota_data_initial | `0x13000` |
| app | `0x20000` |

因为 `CONFIG_PARTITION_TABLE_OFFSET=0xc000`，分区表整体后移了。

启动后串口应该出现：

```
Camera Initialized
Image Captured
person score:13%, no person score 87%
```

### 4.4 模型与算子

`main/person_detect_model_data.cc` 里的模型：

```c
// Keep model aligned to 8 bytes to guarantee aligned 64-bit accesses.
alignas(8) const unsigned char g_person_detect_model_data[] = {
    0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...
};
const int g_person_detect_model_data_len = 300568;
```

- 文件大小约 1.9 MB，因为每个字节都写成了 `0xAB,` 的形式；实际模型 300,568 字节。
- `const` 让数组进 flash 的 `.rodata`；没有 `const` 会进 `.data`，上电复制到 RAM，白占 300 KB。
- `alignas(8)` 是 FlatBuffers 的 8 字节对齐要求。
- 符号名 `g_person_detect_model_data` 必须和头文件、`main_functions.cc` 一致；用 `xxd -i` 重新生成时特别容易改错。

这个工程用到的 5 个算子：

```c
static tflite::MicroMutableOpResolver<5> micro_op_resolver;
micro_op_resolver.AddAveragePool2D();
micro_op_resolver.AddConv2D();
micro_op_resolver.AddDepthwiseConv2D();
micro_op_resolver.AddReshape();
micro_op_resolver.AddSoftmax();
```

如果换模型后用了新算子，`AllocateTensors()` 或 `Invoke()` 会报：

```
Didn't find op for builtin opcode 'CONV_2D' version '3'
```

这时按模型实际算子补 `AddXxx()`，必要时把模板参数 `5` 调大。

模型加载和 schema 检查：

```c
model = tflite::GetModel(g_person_detect_model_data);
if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported version %d.",
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
}
```

### 4.5 Tensor arena 与 ESP-NN scratch

```c
#if CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN
constexpr int scratchBufSize = 60 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif

constexpr int kTensorArenaSize = 100 * 1024 + scratchBufSize;
```

也就是 **100 KB 基础 arena + 60 KB ESP-NN scratch = 163,840 字节**。

分配策略是“先内部 RAM，失败再 PSRAM”：

```c
tensor_arena = (uint8_t *) heap_caps_malloc(
    kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *) heap_caps_malloc(
        kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}
```

实测：

```
INFO arena=120996/163840 input=9216 output=2 out_scale=3906/1e6 out_zp=-128
```

- 请求 163,840 字节，实际用到 120,996 字节；
- 本次 163,840 字节分配成功；代码优先内部 RAM，失败才回退 PSRAM；
- 输入 9,216 字节（96×96×1），输出 2 字节；
- 输出 scale=0.00390625，zero_point=-128。

如果 `AllocateTensors()` 失败，先看 `interpreter->arena_used_bytes()`，再决定加多少；不要一上来就设几百 KB，内部 RAM 还要留给 Wi-Fi、摄像头和任务栈。

### 4.6 推理循环与输出

`main_functions.cc` 的 `loop()`（为了写笔记加了耗时打点）：

```c
void loop() {
  static int64_t last_report = 0;
  static int64_t capture_us = 0;
  static int64_t invoke_us = 0;
  static int frames = 0;

  int64_t t0 = esp_timer_get_time();
  if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels,
                            input->data.int8)) {
    MicroPrintf("Image capture failed.");
  }
  int64_t t1 = esp_timer_get_time();

  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }
  int64_t t2 = esp_timer_get_time();

  capture_us += t1 - t0;
  invoke_us += t2 - t1;
  frames++;
  /* 每秒打印一次 capture / invoke / loop fps，略 */
}
```

`GetImage()` 的灰度路径：

```c
camera_fb_t *fb = esp_camera_fb_get();
if (!fb) {
    ESP_LOGE(TAG, "Camera capture failed");
    return kTfLiteError;
}

MicroPrintf("Image Captured\n");
for (int i = 0; i < image_width * image_height; i++) {
    image_data[i] = ((uint8_t *) fb->buf)[i] ^ 0x80;
}
esp_camera_fb_return(fb);
```

`RespondToDetection()` 把概率转成百分比并打印：

```c
int person_score_int = person_score * 100 + 0.5;
MicroPrintf("person score:%d%%, no person score %d%%",
            person_score_int, 100 - person_score_int);
```

实测串口输出（镜头前无人）：

```
INFO arena=120996/163840 input=9216 output=2 out_scale=3906/1e6 out_zp=-128
PERF capture=0.523ms invoke=49.601ms loop=16fps
person score:13%, no person score 87%
Image Captured

person score:12%, no person score 88%
```

- 无人场景下 person score 在 9%–14% 波动，no person score 在 86%–91%；
- 把镜头对准人时，person score 会明显升高；示例的显示路径以 **60%** 作为“有人”的门槛；
- 二分类也可以直接取两个分数中较大的那个，效果等价于 50% 阈值；60% 是为了减少误报。

### 4.7 实测性能

| 项 | 实测值 |
|---|---:|
| 摄像头取帧 + 转 int8 | 0.52 ms |
| `Invoke()` | **49.6 ms** |
| 循环帧率（含每帧串口打印 + 1 ms delay） | **16 fps** |
| 纯推理理论帧率（1000 / 49.6） | 约 20 fps |
| 模型大小 | 300,568 字节 |
| arena 请求 / 实际使用 | 163,840 / 120,996 字节 |
| app 固件 | 759,024 字节（0xb94f0） |
| app 分区余量 | 61% |

关键结论：

- **瓶颈在推理，不在摄像头**。取帧只占 0.5 ms，`Invoke()` 占了 49.6 ms；
- 每帧打印 `Image Captured` 和分数会拖慢循环；做纯性能测试时，把日志改成每秒一次，帧率会更接近 20 fps；
- 这个数字是 ESP-NN 加速后的结果；如果关掉 ESP-NN 走 TFLM 参考实现，耗时会明显变长（对比方法见 5.7）。

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

### 5.2 为什么只注册 5 个算子？

TFLM 是“按需编译”的思路：只把模型用到的算子链接进来，减小固件体积。person_detect 模型只用了：

- `AVERAGE_POOL_2D`
- `CONV_2D`
- `DEPTHWISE_CONV_2D`
- `RESHAPE`
- `SOFTMAX`

所以 `MicroMutableOpResolver<5>` 就够了。换模型后如果少注册一个，运行时会明确报 `Didn't find op ...`。

### 5.3 int8 和 float32 走的是不同内核

这是最关键的技术点之一。看 `esp-tflite-micro` 的卷积实现 `tensorflow/lite/micro/kernels/esp_nn/conv.cc`：

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

本项目的模型是 int8，输入输出也是 int8，所以走的是 esp-nn 分支。`nm` 可以看到 S3 专用符号：

```bash
nm build/person_detection.elf | grep esp_nn
```

节选：

```
T esp_nn_conv_s8_esp32s3
T esp_nn_conv_s8_3x3_opt
T esp_nn_conv_s8_mult8_1x1_esp32s3
（depthwise / pooling / softmax 也有 S3 专用实现）
```

ESP32-S3 有专门的汇编/SIMD 路径；ESP32-C5 更多走通用优化 C，所以同一个模型在 C5 上通常没有 S3 这种加速幅度。

### 5.4 为什么固件里还有 esp-nn？

因为 `esp-tflite-micro` 在组件清单里依赖 `esp-nn`；构建时它会把原来的 `conv.cc`、`pooling.cc`、`fully_connected.cc`、`softmax.cc` 等替换成带 esp-nn 分支的版本。这个版本同时包含 int8 和 float 分支，所以库被链接进来，但只有 int8 模型会走优化内核。

### 5.5 怎么确认有没有真的调用？

- **看代码分支**：最直接；int8 输入 → `esp_nn_conv_s8`。
- **看符号**：`nm build/person_detection.elf | grep esp_nn` 只能证明“链接了”，不能证明“执行了”。
- **看耗时**：在 `Invoke()` 前后用 `esp_timer_get_time()` 打点，对比 int8 和 float 两个模型。
- **看 profiling**：接 TFLM 的 `MicroProfiler`（在解释器构造时传入）。

本项目的 49.6 ms 是 int8 + ESP-NN 的实际结果；模型输入张量是 int8，所以可以确定走的是 int8 分支。

### 5.6 ESP32-S3 上的 esp-nn 优化程度

esp-nn 的汇编/SIMD 专用路径主要覆盖 **ESP32-S3** 和 **ESP32-P4**。S3 的卷积、深度卷积、池化等都有专用实现，这是 person_detect 能在 50 ms 内跑完 96×96 模型的重要原因。

对比：

| 芯片 | esp-nn 路径 | 同一模型的预期 |
|---|---|---|
| ESP32-S3 | 汇编/SIMD 优化 | 最快，本项目 49.6 ms |
| ESP32-P4 | 汇编/SIMD 优化（RISC-V PIE） | 更快，但平台不同 |
| ESP32-C5 | 通用优化 C / ANSI C | 能用，但加速幅度小 |
| ESP32 | 通用优化 C | 更慢 |

### 5.7 想自己量化 ESP-NN 的收益

复制工程，关掉 ESP-NN，重新构建：

```bash
cp -a tflite_person tflite_person_ref
cd tflite_person_ref
rm -rf build
sed -i 's/^CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN=y/# CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN is not set/' sdkconfig
idf.py build
idf.py -p /dev/ttyACM0 flash
```

然后对比串口里的 `PERF ... invoke=...`。

注意：

- 关掉后 `scratchBufSize` 变成 0，arena 从 160 KB 降到 100 KB；
- 参考实现仍然能跑，只是慢；
- 不要只看固件大小判断性能，必须看 `Invoke()` 耗时；
- 本次参考构建已生成（app 0xaf470），但板子 USB 在烧录时掉线，未取得完整对比数据；上面的命令可复现。

---

## 6. XIAO ESP32-S3 平台配置

### 6.1 芯片与工程参数

| 项 | 值 |
|---|---|
| 目标 | `esp32s3` |
| CPU | 双核 Xtensa LX7，240 MHz |
| Flash | 8 MB，80 MHz |
| PSRAM | 8 MB，Octal，80 MHz |
| 控制台 | USB Serial/JTAG，`/dev/ttyACM0` |
| 分区表偏移 | `0xc000` |
| app 分区 | `ota_0` / `ota_1`，各 `0x1E0000`（1.875 MB） |
| 工程名 | `person_detection` |

启动日志（节选）：

```
I (186) octal_psram: vendor id    : 0x0d (AP)
I (186) octal_psram: density      : 0x03 (64 Mbit)
I (186) octal_psram: Readlatency  : 0x02 (10 cycles@Fixed)
```

### 6.2 Flash 与分区布局

`partitions.csv`：

```csv
# Name,   Type, SubType, Offset,   Size,      Flags
nvs,      data, nvs,     0xd000,   0x6000,
otadata,  data, ota,     ,         0x2000,
phy_init, data, phy,     ,         0x1000,
ota_0,    app,  ota_0,   0x20000,  0x1E0000,
ota_1,    app,  ota_1,   ,         0x1E0000,
```

实际偏移：

```
bootloader        0x0
partition-table   0xc000
ota_data_initial  0x13000
app               0x20000
```

两个 app 槽各 1.875 MB，当前固件 759,024 字节，余量 61%。

### 6.3 PSRAM

- 摄像头帧缓冲放 PSRAM：`config.fb_location = CAMERA_FB_IN_PSRAM`。
- tensor arena 优先内部 RAM，内部不够才用 PSRAM；PSRAM 延迟更高，推理会变慢。
- `CONFIG_SPIRAM_RODATA` 没有开，模型仍然在 flash 的 `.rodata`，不会复制到 PSRAM。
- 可以从启动日志确认 PSRAM 容量和模式；看不到 `octal_psram` 日志，通常是 `CONFIG_SPIRAM` 没开或模式选错。

### 6.4 控制台与无 TTY 环境

本工程用原生 USB Serial/JTAG：

```
CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y
CONFIG_ESP_CONSOLE_SECONDARY_NONE=y
```

Linux 下是 `/dev/ttyACM0`。

`idf.py monitor` 要求标准输入是 TTY；在 CI、脚本或某些 IDE 终端里会报：

```
Error: Monitor requires standard input to be attached to TTY.
```

这时用 pyserial 直接读：

```python
import serial, time

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.5)
end = time.time() + 15
while time.time() < end:
    data = ser.read(8192)
    if data:
        print(data.decode('utf-8', 'replace'), end='')
ser.close()
```

### 6.5 构建、烧录与 VMware 掉线

正常流程：

```bash
. $IDF_PATH/export.sh
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyACM0 flash
```

如果在 VMware 里烧录时 `/dev/ttyACM0` 消失、`lsusb` 看不到 `303a:1001`，通常是 USB 重新枚举失败。处理顺序：

1. 关闭所有占用串口的 `idf.py monitor` / 串口助手；
2. 重新插拔 USB，确认 `lsusb` 出现 `303a:1001`；
3. 如果板子已经在下载模式，用 esptool 的 `--before=no_reset` 直接写：

```bash
python -m esptool --chip esp32s3 -p /dev/ttyACM0 -b 460800 \
  --before=no_reset --after=hard_reset write_flash \
  --flash_mode dio --flash_freq 80m --flash_size 8MB \
  0x0 build/bootloader/bootloader.bin \
  0xc000 build/partition_table/partition-table.bin \
  0x13000 build/ota_data_initial.bin \
  0x20000 build/person_detection.bin
```

4. 仍然不行就换 USB 口/线，或让板子重新上电。

### 6.6 SD 卡与 LED 的冲突

本文不用 SD，但如果你要加 SD 卡功能：

- 板子出厂时 SD 的 CS 是 **GPIO21**，和用户 LED 共用；
- 某些 BSP（例如 `espp/xiao-esp32s3-sense`）里写的是 GPIO3，和实物不符；
- 挂载 SD 后 GPIO21 被 SPI 占用，用户 LED 就不能再当普通输出；
- 硬件上可以通过 R11/R12 改线，但需要动烙铁；本文不展开。

---

## 7. 性能与量化

### 7.1 模型规模

| 项 | 数值 |
|---|---:|
| 模型大小 | 300,568 字节（约 293.5 KiB） |
| 输入 | 96×96×1 int8 = 9,216 字节 |
| 输出 | 2 × int8 = 2 字节 |
| arena | 请求 163,840 字节，实用 120,996 字节 |
| 推理耗时 | 49.6 ms |
| 循环帧率 | 16 fps（含日志） |

### 7.2 实测性能

| 项 | 值 |
|---|---:|
| capture（取帧 + 转 int8） | 0.52 ms |
| invoke（ESP-NN） | 49.6 ms |
| loop（含串口日志 + 1 ms delay） | 16 fps |
| 纯推理上限 | 约 20 fps |
| app 固件 | 759,024 字节 |
| flash `.rodata` | 378,744 字节（其中模型约 300 KB） |

`Invoke()` 占了循环时间的 99%，所以优化顺序应该是：

1. 换更小的模型 / 更小的输入；
2. 确认 int8 + ESP-NN 真的生效；
3. 减少每帧串口打印；
4. 最后才考虑摄像头分辨率（本 demo 已经是 96×96）。

### 7.3 int8 vs float32

| | int8（本项目） | float32 |
|---|---|---|
| 权重/激活 | 1 字节/个 | 4 字节/个 |
| 运算 | 整数乘加 | 浮点 |
| esp-nn | 走 `esp_nn_*_s8` | 走 `reference_ops`，不加速 |
| 速度 | 快 | 慢 |
| 精度 | 需校准，可能略降 | 通常略好 |
| 部署难度 | 需要 representative dataset | 简单 |
| 适合 | MCU、实时 | 原型、精度优先 |

### 7.4 输入量化与输出反量化

输入：

```c
// 摄像头灰度 0..255 → int8 -128..127
image_data[i] = ((uint8_t *) fb->buf)[i] ^ 0x80;
```

这只是一个位映射；真实数值由 `input->params.scale / zero_point` 决定，不要假设固定是 1/255 和 -128。自定义模型要按张量参数算：

```c
float real = pixel / 255.0f;
int8_t q = (int8_t)(real / input->params.scale + input->params.zero_point);
```

输出：

```c
int8_t raw = output->data.int8[kPersonIndex];
float score = (raw - output->params.zero_point) * output->params.scale;
```

实测 `scale=0.00390625`、`zero_point=-128`，所以 `raw=-128` 对应 0，`raw=127` 对应约 0.996。

### 7.5 什么时候该量化？

- 模型是 float32，但 `Invoke()` 已经接近实时上限；
- Flash/RAM 吃紧；
- 想用 esp-nn 的 int8 内核；
- 想省电。

量化不是“无脑压缩”：校准集要覆盖真实摄像头画面（光照、背景、人距），量化后要用混淆矩阵验证，尤其是“无人”被误判成“有人”的误报率。

### 7.6 本地推理和网络推流的关系

这个 demo **完全不联网**：摄像头 → 模型 → 串口，推理不依赖 Wi-Fi 带宽。

之前摄像头网页推流项目里，QVGA MJPEG 只有 12–18 fps，而摄像头单拍能到 50 fps 以上，瓶颈在 Wi-Fi/TCP 链路；那是“把画传出去”的问题，不是“模型能不能跑”的问题。

如果一边推流一边推理：

- 摄像头、PSRAM 带宽、CPU 会被两个任务共享；
- 推流卡顿不代表推理慢，推理慢也不代表网络差；
- 要分别测量 `capture`、`invoke`、网络发送耗时，不要用一个端到端帧率下结论。

---

## 8. 实战踩坑（重点）

### 8.1 摄像头引脚选不到 GPIO48

**症状**：menuconfig 的 Custom Camera Pinout 里，Y9 最大只能选到 39，选不到 XIAO 的 GPIO48。

**原因**：`main/Kconfig.projbuild` 里 `range` 上限是 39（部分选项 33），Kconfig 不允许输入范围外的值。

**解决**：把 `range` 改成 `0 48`（PWDN/RESET 用 `-1 48`），执行 `idf.py reconfigure`，再进 menuconfig。

### 8.2 摄像头初始化失败

**症状**：

```
E (xxx) app_camera: Camera init failed with error 0x105
```

或一直 `Camera capture failed`。

**原因**：

- 引脚表和实物不一致（尤其是 D0–D7 顺序）；
- XCLK 频率不合适（示例默认 15 MHz，XIAO 用 20 MHz 更稳）；
- PWDN/RESET 被配成了实际不存在的 GPIO；
- SCCB 的 SDA/SCL 接反；
- 摄像头排线没插紧。

**解决**：逐项对照 4.2 的表；先只改 XCLK 到 20 MHz；确认 `PWDN/RESET=-1`；用万用表/原理图核对 SCCB。

### 8.3 app 分区不够

**症状**：

```
Error: app partition is too small for binary ...
```

**原因**：默认 1 MB 单 app 分区，固件 + 模型很容易接近上限；带 OTA 时还要两个槽。

**解决**：用 8 MB flash + 自定义分区表，两个 1.875 MB 的 OTA 槽；本项目 app 759 KB，余量 61%。改分区表后注意 `CONFIG_PARTITION_TABLE_OFFSET` 和烧录偏移一起变。

### 8.4 PSRAM 没开，摄像头帧缓冲分配失败

**症状**：`esp_camera_init()` 失败，或启动时 heap 不足。

**原因**：`config.fb_location = CAMERA_FB_IN_PSRAM`，但 `CONFIG_SPIRAM` 没开，或者 PSRAM 模式选错（XIAO 是 Octal，不是 Quad）。

**解决**：打开 `CONFIG_SPIRAM`、`CONFIG_SPIRAM_MODE_OCT`、`CONFIG_SPIRAM_SPEED_80M`；看启动日志有没有 `octal_psram: density 0x03 (64 Mbit)`。

### 8.5 `CONFIG_NN_OPTIMIZED` 和 `CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN` 不是一回事

**症状**：以为开了 ESP-NN，但 `scratchBufSize` 还是 0，arena 偏小，或者性能没变化。

**原因**：代码里的条件编译用的是 `CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN`；`CONFIG_NN_OPTIMIZED` 是 esp-nn 自己的优化开关，两个都要对。

**解决**：

```c
#if CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN
constexpr int scratchBufSize = 60 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
```

确认 `sdkconfig` 里 `CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN=y`、`CONFIG_NN_OPTIMIZED=y`、`CONFIG_NN_OPTIMIZATIONS=1`。

### 8.6 arena 太小 / 太大

**症状**：`AllocateTensors() failed`，或推理时崩溃。

**原因**：中间张量总需求超过 `kTensorArenaSize`；或者内部 RAM 不够，代码回退到 PSRAM 后变慢。

**解决**：打印 `interpreter->arena_used_bytes()`；本项目实用 120,996 字节，请求 163,840 字节。先按实用值加 10–20% 余量，不要盲目设 1 MB。

### 8.7 `xxd -i` 生成的文件丢了 `const` / `alignas`

**症状**：模型占用 300 KB RAM，链接失败，或推理时对齐异常。

**原因**：`xxd -i` 默认生成非 `const` 数组；直接覆盖会丢掉仓库里手工加的 `alignas(8)` 和 `const`。

**解决**：替换时保留：

```c
alignas(8) const unsigned char g_person_detect_model_data[] = { ... };
const int g_person_detect_model_data_len = 300568;
```

并确认符号名和头文件一致。

### 8.8 误以为模型在 RAM

**症状**：`nm` 看到符号类型是 `D`，以为模型在 `.data` / RAM。

**真相**：ESP32-S3 的 flash 数据映射也在 `0x3C...` 段；`nm` 的 `D` 不一定代表内部 RAM。用 readelf 看 section 才准确：

```bash
xtensa-esp32s3-elf-readelf -sW build/person_detection.elf | grep g_person_detect_model_data
# 4708: 3c05e840 0x49618 OBJECT GLOBAL DEFAULT 15 g_person_detect_model_data
xtensa-esp32s3-elf-readelf -SW build/person_detection.elf | grep flash.rodata
# [15] .flash.rodata PROGBITS 3c050120 ... 05c778
```

`idf.py size` 里 `Flash Data 379,000`、`.rodata 378,744` 也说明模型在 flash。

### 8.9 输入/输出类型搞错

**症状**：模型能跑，但分数乱飞，或者永远 50%。

**原因**：

- 模型是 int8，代码却用 `input->data.f`；
- 输入没有按 zero_point 映射；
- 输出用 `uint8` 读，却忘了减 zero_point；
- 自定义模型的输入量化参数和官方模型不同，却照抄 `^0x80`。

**解决**：先打印：

```c
MicroPrintf("input type=%d scale=%d/1e6 zp=%d",
            input->type,
            (int)(input->params.scale * 1000000),
            input->params.zero_point);
```

再按实际参数做量化/反量化。二分类的 `data.uint8` 只是别名，建议统一用 `data.int8`。

### 8.10 类别索引反了

**症状**：人走过来显示 “no person”，没人显示 “person”。

**原因**：`kPersonIndex` / `kNotAPersonIndex` 和训练标签顺序不一致。本模型是：

```c
constexpr int kPersonIndex = 1;
constexpr int kNotAPersonIndex = 0;
```

**解决**：换模型后先确认输出张量的类别顺序；训练时的标签顺序、`kCategoryLabels[]`、端侧索引三者必须一致。

### 8.11 `CLI_ONLY_INFERENCE` / `DISPLAY_SUPPORT` 走错分支

**症状**：没有摄像头初始化，或者程序停在 CLI；或者接了屏却走灰度。

**原因**：

- `CLI_ONLY_INFERENCE` 被手动定义，`InitCamera()` 直接跳过；
- `DISPLAY_SUPPORT` 由 BSP 决定，没接屏时不应该定义；
- `CONFIG_TFLITE_USE_BSP_NONE=y` 时 `DISPLAY_SUPPORT` 不会打开。

**解决**：本项目在 S3 + 摄像头的组合下，`CLI_ONLY_INFERENCE` 不定义，走 `loop()` 摄像头路径；不要为了用 `detect_image` 命令手动打开它，否则摄像头不会初始化。

### 8.12 任务栈溢出 / 看门狗

**症状**：启动后 panic、栈溢出，或长时间不喂狗。

**原因**：摄像头初始化、TFLM、CLI 共用 `tf_main` 栈；4 KB 不够。

**解决**：`tf_main` 栈设为 8 KB；`loop()` 末尾 `vTaskDelay(1)`；如果加了自己的大局部数组，再适当加大栈或改成静态/堆分配。

### 8.13 USB Serial/JTAG 在 VMware 里掉线

**症状**：`idf.py flash` 报 `Could not configure port`、`/dev/ttyACM0` 消失、`lsusb` 看不到设备。

**原因**：烧录触发 USB 重新枚举，VMware 没把新设备接回虚拟机；或者 `idf.py monitor` 还占着端口。

**解决**：关掉 monitor/串口助手；重新插拔；用 6.5 的 `--before=no_reset` esptool 命令；必要时换 USB 口/线。

### 8.14 串口日志拖慢帧率

**症状**：`invoke` 只有 49.6 ms，但 loop 只有 16 fps，不是 20 fps。

**原因**：每帧打印 `Image Captured` 和 `person score...`，USB Serial/JTAG 输出本身要时间；还有 `vTaskDelay(1)`。

**解决**：性能测试时把日志改成每秒一次，或只统计不打印；不要用带日志的帧率代表模型极限。

### 8.15 SD 卡和 LED 冲突

**症状**：挂载 SD 后用户 LED 不亮/不受控。

**原因**：GPIO21 同时是 SD CS 和用户 LED；挂载后 GPIO21 被 SPI 驱动占用。

**解决**：本项目不用 SD；要同时用，需要改硬件（R11/R12）或接受 LED 不可用。不要直接把 LED 当普通 GPIO 拉高，可能干扰 SD 通信。

### 8.16 把“人形检测”当成了“人脸识别”

**症状**：期望模型输出“这是谁”或人脸框，实际只有两个分数。

**原因**：person_detection 是整图二分类，不是人脸检测/识别。

**解决**：

- 只要“有没有人” → 本项目够用；
- 要“有没有人脸” → 训练 face / non-face 二分类，流程同第 9 章；
- 要人脸框 → 用目标检测模型（如 ESP-DL 的 HumanFaceDetect）；
- 要“是谁” → 用 embedding + 相似度（如 ESP-DL 的 HumanFaceRecognizer），不是二分类。

---

## 9. 换成自己的二分类模型（人脸 / 非人脸）

> 本章是“把官方 person 模型换成自己的二分类模型”的通用流程。官方模型是 person / no-person；你要做人脸 / 非人脸，只需要把数据和标签换掉，代码骨架不变。

### 9.1 先确认任务类型

| 目标 | 模型类型 | 输出 | 本文是否覆盖 |
|---|---|---|---|
| 有没有人 | 二分类 | 2 个分数 | 是（官方模型） |
| 有没有人脸 | 二分类 | 2 个分数 | 是（换数据/标签） |
| 人脸在画面哪里 | 目标检测 | 框 + 分数 | 否，需检测模型 |
| 这是谁 | 人脸识别 | embedding / 相似度 | 否，需识别模型 |

**二分类只能回答“像不像”，不能回答“是谁”。** 如果你的需求是门禁/身份，别用二分类硬凑。

### 9.2 数据

- 输入统一为 **96×96 灰度**（和部署端一致）；如果模型输入改成别的尺寸，端侧 `kNumCols/kNumRows` 也要改。
- 正类：人脸（不同人、光照、角度、距离、遮挡）。
- 负类：背景、天空、墙面、物体、模糊画面、部分人脸。
- 类别要平衡；负类要覆盖真实场景，否则模型会把“像人脸的背景”误判成正类。
- 按人/场景/时间划分训练集和测试集，不要随机打乱同一段视频的相邻帧，否则准确率会虚高。
- 数据量：每类几百到几千张起步；先跑通再扩。

采图可以复用之前的摄像头 Web demo（`/camera` 返回 JPEG），或者把 JPEG 存到 SD 卡再拷到 PC。PC 端统一转灰度、缩放到 96×96：

```python
import cv2

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (96, 96))
```

正负类分目录存放；文件名不要带中文/空格，后面生成 CSV 或 `ImageDataGenerator` 更省事。

### 9.3 训练一个小 CNN

Keras 示例（结构可按需调整）：

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input((96, 96, 1)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.DepthwiseConv2D(3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(32, 1, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=30, batch_size=64)
```

标签顺序要和端侧一致，例如：

```python
# 0 = 非人脸，1 = 人脸
y = np.array([0, 1, ...])
```

注意：`GlobalAveragePooling2D` 在 TFLite 里可能变成 `Mean` 算子；`Dense` 会变成 `FullyConnected`。部署前用 `tf.lite.Interpreter` 看实际算子，按需在 `MicroMutableOpResolver` 里注册。

### 9.4 转 int8 TFLite

```python
def representative_dataset():
    for image in calib_images:          # 100~500 张真实灰度图
        x = image.astype(np.float32) / 255.0
        yield [x.reshape(1, 96, 96, 1)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open('face.tflite', 'wb') as f:
    f.write(tflite_model)
```

转换后在 PC 上验证：

```python
interp = tf.lite.Interpreter(model_path='face.tflite')
interp.allocate_tensors()
print(interp.get_input_details())
print(interp.get_output_details())
```

确认：

- 输入形状 `[1, 96, 96, 1]`、dtype `int8`；
- 输出形状 `[1, 2]`、dtype `int8`；
- 输入的 `scale` / `zero_point`（不要假设一定是 1/255 和 -128）；
- 用几张实拍图跑一遍，和 Keras 的浮点输出对比。

### 9.5 端侧替换

1. 生成 C 数组：

   ```bash
   xxd -i -n g_person_detect_model_data face.tflite > person_detect_model_data.cc
   ```

   然后在文件里补上 `alignas(8) const`，并保留 `g_person_detect_model_data_len`；注意 `xxd` 生成的长度变量类型通常是 `unsigned int`，而官方头文件声明的是 `const int`，两边要统一，否则会编译冲突。

2. 更新 `main/model_settings.h`：

   ```c
   constexpr int kNumCols = 96;
   constexpr int kNumRows = 96;
   constexpr int kNumChannels = 1;
   constexpr int kCategoryCount = 2;
   constexpr int kPersonIndex = 1;       // 人脸
   constexpr int kNotAPersonIndex = 0;   // 非人脸
   ```

   如果你不想改语义，也可以保留 `person` 命名，把“人脸”当正类。

3. 更新标签 `main/model_settings.cc`：

   ```c
   const char* kCategoryLabels[kCategoryCount] = {
       "notface",
       "face",
   };
   ```

4. 更新算子注册：按 `face.tflite` 实际用到的算子，在 `main_functions.cc` 里 `AddXxx()`；不要只照抄 5 个。

5. 调整输入量化：如果新模型的输入 `scale/zero_point` 和官方模型不同，把 `image_provider.cc` 里的 `^0x80` 换成按参数计算的版本。

6. 调整输出反量化：用 `output->params.scale` / `zero_point`，不要写死 1/256。

7. 调 arena：先保留 `100KB + 60KB`，启动后看 `arena_used_bytes()`，再增减。

8. 改阈值：二分类可以直接 argmax，也可以按业务设阈值（例如人脸分数 > 0.6 才触发）。

### 9.6 模型更新方式

| 方式 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| C 数组 | `xxd -i` 后编进固件 | 最简单，启动即可用 | 换模型要重编/重烧整个固件 |
| `EMBED_FILES` | CMake 直接嵌入 `.tflite` | 不生成巨大 C 文件，编译快 | 仍然跟固件一起烧 |
| 独立 flash 分区 | 模型写到 `model` 分区，`esp_partition_mmap` 映射 | 模型/app 分离，可单独更新 | 要改分区表、加载逻辑、对齐 |
| SD / 文件系统 | 读进 RAM/PSRAM 后 `GetModel` | 换模型最灵活 | 要挂载文件系统，SD 还有 GPIO21 冲突 |
| 网络下载 | 下到 PSRAM 再推理 | 远程更新 | 需要网络和校验，实现复杂 |

外置模型的共同要求：

- 缓冲区必须对齐（至少 8 字节，FlatBuffers 要求）；
- 模型缓冲区在推理期间不能释放/复用；
- 校验长度和 schema 版本；
- 从 SD/文件系统读取时，先完整读入连续内存，不能把 `FILE*` 直接传给 `GetModel`。

### 9.7 如果你真的要做“人脸识别”

二分类模型做不到“识别是谁”。常见做法：

1. **人脸检测**：先定位人脸框（ESP-DL `HumanFaceDetect` 或目标检测模型）。
2. **人脸对齐/裁剪**：把人脸缩放到识别模型输入尺寸。
3. **特征提取**：用 embedding 模型输出特征向量。
4. **比对**：和注册库里的特征算余弦相似度，超过阈值判为同一个人。

这条路线是“检测 + 识别”，不是二分类；P4 分支的 esp-dl 笔记里有 `HumanFaceDetect` / `HumanFaceRecognizer` 的用法和坑。XIAO ESP32-S3 也能跑 esp-dl，但模型和内存预算要另算。

### 9.8 精度验证

- 用混淆矩阵，不要只看总准确率；
- 重点看误报（负类被判成正类）和漏报（正类被判成负类）；
- 阈值按业务调：门禁宁可漏报少误报，唤醒类应用可能相反；
- 校准集要来自真实摄像头，不要用训练集图片；
- 量化前后各测一次，确认精度下降可接受。

---

## 10. 学习路线与速查命令

### 10.1 建议学习路线

| 阶段 | 目标 | 关键内容 |
|---|---|---|
| 0 | 搞清概念 | TinyML、TFLM、esp-nn、esp-dl 的层级关系 |
| 1 | 跑通示例 | `idf.py build flash`，看到 `Camera Initialized` 和分数 |
| 2 | 理解输入 | 96×96 灰度、`^0x80`、int8 量化参数 |
| 3 | 理解模型 | 5 个算子、2 类输出、反量化、阈值 |
| 4 | 理解内存 | arena、ESP-NN scratch、内部 RAM vs PSRAM |
| 5 | 换模型 | 训练 face/non-face、转 int8、替换 C 数组 |
| 6 | 优化 | 关日志、调 arena、对比 ESP-NN、换输入尺寸 |
| 7 | 工程化 | 模型分区/OTA、异常处理、版本管理 |

### 10.2 速查命令

```bash
# 环境
. $IDF_PATH/export.sh

# 目标与构建
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyACM0 flash

# 菜单配置
idf.py menuconfig
# → Application Configuration → Camera Configuration

# 查看大小
idf.py size
idf.py size-components

# 查看模型符号和 ESP-NN 符号
nm build/person_detection.elf | grep g_person_detect_model_data
nm build/person_detection.elf | grep esp_nn

# 查看模型所在 section
xtensa-esp32s3-elf-readelf -sW build/person_detection.elf | grep g_person_detect_model_data
xtensa-esp32s3-elf-readelf -SW build/person_detection.elf | grep flash.rodata

# 生成模型 C 数组
xxd -i -n g_person_detect_model_data model.tflite > person_detect_model_data.cc

# 手动烧录（板子已在下载模式 / ttyACM0 掉线时）
python -m esptool --chip esp32s3 -p /dev/ttyACM0 -b 460800 \
  --before=no_reset --after=hard_reset write_flash \
  --flash_mode dio --flash_freq 80m --flash_size 8MB \
  0x0 build/bootloader/bootloader.bin \
  0xc000 build/partition_table/partition-table.bin \
  0x13000 build/ota_data_initial.bin \
  0x20000 build/person_detection.bin
```

### 10.3 关键数字

| 项 | 值 |
|---|---:|
| 输入 | 96×96×1 int8 |
| 模型 | 300,568 字节 |
| 量化 | 全 int8；输出 scale=1/256，zp=-128 |
| arena | 请求 163,840，实用 120,996 字节 |
| 取帧 | 0.52 ms |
| 推理 | 49.6 ms |
| 循环帧率 | 16 fps（含日志） |
| app | 759,024 字节 |
| 摄像头 XCLK | 20 MHz |
| 摄像头数据引脚 | D0–D7 = 15,17,18,16,14,12,11,48 |
| 用户 LED / SD CS | GPIO21（冲突） |

### 10.4 无 TTY 时读串口

```python
import serial, time

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.5)
end = time.time() + 15
buf = bytearray()
while time.time() < end:
    d = ser.read(8192)
    if d:
        buf += d
ser.close()
print(buf.decode('utf-8', 'replace'))
```

---

## 11. 参考

- esp-tflite-micro：<https://github.com/espressif/esp-tflite-micro>
- esp-nn：<https://github.com/espressif/esp-nn>
- esp32-camera：<https://github.com/espressif/esp32-camera>
- TFLite Micro 文档：<https://ai.google.dev/edge/litert/microcontrollers/overview>
- ESP-IDF 编程指南：<https://docs.espressif.com/projects/esp-idf/>
- XIAO ESP32S3 Sense 入门：<https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/>
- person_detection 示例：<https://github.com/espressif/esp-tflite-micro/tree/master/examples/person_detection>
- 公共基础：[model-deployment.md](https://github.com/yh-xh/ESP32-TinyML/blob/main/model-deployment.md)、[neural-networks.md](https://github.com/yh-xh/ESP32-TinyML/blob/main/neural-networks.md)
- 配套笔记：[esp32-p4（esp-dl）](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-p4)、[esp32-c5（esp-nn）](https://github.com/yh-xh/ESP32-TinyML/tree/esp32-c5)

---

> 本文记录来自 XIAO ESP32S3 Sense 上实际移植 `person_detection` 的调试，欢迎提 Issue 交流或补充。
