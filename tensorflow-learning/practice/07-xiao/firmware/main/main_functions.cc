/* TFLM + esp-nn 推理示例：16x16 横条/竖条 int8 CNN。
 *
 * 模型来自 practice/06-tflite/image_cnn_int8.tflite，
 * 用 xxd -i 转成 main/model.cc 里的 g_model 数组。
 *
 * 这个示例不接摄像头，而是在固件里生成两张 16x16 测试图：
 * - 横条：中间两行是 1，其余是 0；
 * - 竖条：中间两列是 1，其余是 0。
 *
 * 每 500ms 跑一次推理，串口打印：
 *   truth=... p(vertical)=... pred=...
 */

#include <cmath>
#include <cstdint>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

#include "main_functions.h"
#include "model.h"

namespace {

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// S3 上这个 CNN 很小，32KB arena 足够。
// 太小会在 AllocateTensors() 时报错，太大浪费内部 RAM。
constexpr int kTensorArenaSize = 32 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

constexpr int kHeight = 16;
constexpr int kWidth = 16;
int inference_count = 0;

// 在输入张量里画一张 16x16 的图。
// pixel 是 float 的 0/1，写进 int8 张量前要先量化：
//   q = round(pixel / scale) + zero_point
void fill_image(bool vertical) {
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      bool bar = vertical ? (x >= 7 && x <= 8) : (y >= 7 && y <= 8);
      float pixel = bar ? 1.0f : 0.0f;

      int32_t q = static_cast<int32_t>(
                      std::round(pixel / input->params.scale)) +
                  input->params.zero_point;
      if (q < -128) q = -128;
      if (q > 127) q = 127;

      // 输入 shape 是 (1, 16, 16, 1)，batch=1，
      // 所以 256 个像素直接按行优先写进去。
      input->data.int8[y * kWidth + x] = static_cast<int8_t>(q);
    }
  }
}

void run_inference(bool vertical) {
  fill_image(vertical);

  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  // 输出是 int8，不是概率。先反量化：
  //   p = (q - zero_point) * scale
  float p = (output->data.int8[0] - output->params.zero_point) *
            output->params.scale;

  // sigmoid 二分类：p 是“竖条”的概率。
  const char* truth = vertical ? "vertical" : "horizontal";
  const char* pred = (p >= 0.5f) ? "vertical" : "horizontal";
  MicroPrintf("truth=%-10s p(vertical)=%.4f pred=%s", truth, p, pred);
}

}  // namespace

void setup() {
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema version %d, expected %d",
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // 模型里用到了哪些算子，这里就要注册哪些。
  // 06 的 int8 模型算子：
  //   CONV_2D / MAX_POOL_2D / RESHAPE / FULLY_CONNECTED / LOGISTIC
  static tflite::MicroMutableOpResolver<5> resolver;
  if (resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddMaxPool2D() != kTfLiteOk ||
      resolver.AddReshape() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddLogistic() != kTfLiteOk) {
    MicroPrintf("Failed to register ops");
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed; arena may be too small");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  MicroPrintf("input type=%d shape=[%d,%d,%d,%d]",
              static_cast<int>(input->type),
              input->dims->data[0], input->dims->data[1],
              input->dims->data[2], input->dims->data[3]);
  MicroPrintf("input scale=%.8f zero_point=%d",
              input->params.scale, input->params.zero_point);

  MicroPrintf("output type=%d shape=[%d,%d]",
              static_cast<int>(output->type),
              output->dims->data[0], output->dims->data[1]);
  MicroPrintf("output scale=%.8f zero_point=%d",
              output->params.scale, output->params.zero_point);

  MicroPrintf("tensor arena used: %d bytes",
              static_cast<int>(interpreter->arena_used_bytes()));
  MicroPrintf("model bytes: %d", g_model_len);

  inference_count = 0;
}

void loop() {
  // 交替测试横条和竖条。
  run_inference((inference_count % 2) == 1);
  inference_count++;
}
