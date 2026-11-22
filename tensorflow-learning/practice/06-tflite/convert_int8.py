"""把 05 训练好的 CNN 转成 int8 TFLite，并在 PC 上验证。

这一节做四件事：
1. 加载 05-cnn/image_cnn.keras；
2. 转一份 float32 .tflite 作为基准；
3. 用 representative dataset 转一份 int8 .tflite；
4. 用 tf.lite.Interpreter 对比两者的输入输出、概率和准确率。

Run:
    conda activate tf-esp
    cd practice/06-tflite
    python convert_int8.py

如果提示找不到 image_cnn.keras，先运行：
    cd practice/05-cnn
    python image_cnn.py
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "05-cnn" / "image_cnn.keras"
FLOAT_PATH = BASE_DIR / "image_cnn_float32.tflite"
INT8_PATH = BASE_DIR / "image_cnn_int8.tflite"

if not MODEL_PATH.exists():
    raise SystemExit(
        f"找不到 {MODEL_PATH}\n"
        "先运行 practice/05-cnn/image_cnn.py 生成模型。"
    )

# 加载 05 训练好的 Keras 模型。
# 这里只做推理和转换，不再训练。
model = tf.keras.models.load_model(MODEL_PATH)
print("loaded model:", MODEL_PATH)
print("parameters:", model.count_params())


def make_images(n, rng):
    """生成和 05 一样的 16x16 横条/竖条图，用于校准和测试。

    images: (n, 16, 16, 1), float32, 取值 0/1
    labels: (n,), int32, 0=横条，1=竖条
    """
    images = np.zeros((n, 16, 16, 1), dtype=np.float32)
    labels = rng.integers(0, 2, size=n).astype(np.int32)

    for i in range(n):
        thickness = int(rng.integers(2, 4))
        if labels[i] == 0:
            row = int(rng.integers(0, 16 - thickness + 1))
            images[i, row:row + thickness, :, 0] = 1.0
        else:
            col = int(rng.integers(0, 16 - thickness + 1))
            images[i, :, col:col + thickness, 0] = 1.0

    return images, labels


# representative dataset 用来估计输入/输出的数值范围。
# 量化器会拿这些样本跑一遍模型，统计每一层的 min/max，
# 再决定 int8 的 scale 和 zero_point。
#
# 它不需要标签，只要输入样本；这里每次给 1 张图。
def representative_dataset():
    rng = np.random.default_rng(123)
    for _ in range(100):
        images, _ = make_images(1, rng)
        # Keras 模型只有一个输入，所以 yield 一个列表：
        # [images] 里的 images shape 是 (1, 16, 16, 1)。
        yield [images]


# 1. 先转 float32 TFLite，作为对比基准。
# 这一步不量化，输入输出仍然是 float32。
converter = tf.lite.TFLiteConverter.from_keras_model(model)
float_model = converter.convert()
FLOAT_PATH.write_bytes(float_model)
print(f"\nfloat32 tflite: {FLOAT_PATH} ({len(float_model)} bytes)")

# 2. 转 int8 TFLite。
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 启用默认优化；这里主要是训练后量化。
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 提供校准数据。没有它，int8 量化不知道数值范围。
converter.representative_dataset = representative_dataset

# 限制只使用 TFLite 内置的 int8 算子。
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# 让模型的输入和输出也变成 int8。
# 如果保持 float32，TFLM 端还要自己做 float/int8 转换。
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

int8_model = converter.convert()
INT8_PATH.write_bytes(int8_model)
print(f"int8 tflite:    {INT8_PATH} ({len(int8_model)} bytes)")


def make_interpreter(model_bytes):
    """用模型内容创建解释器，并分配输入/输出张量。"""
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    return interpreter


def describe(interpreter, name):
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    print(f"\n{name} input:")
    print("  shape:", input_detail["shape"])
    print("  dtype:", input_detail["dtype"])
    print("  quantization (scale, zero_point):", input_detail["quantization"])
    print(f"{name} output:")
    print("  shape:", output_detail["shape"])
    print("  dtype:", output_detail["dtype"])
    print("  quantization (scale, zero_point):", output_detail["quantization"])


float_interp = make_interpreter(float_model)
int8_interp = make_interpreter(int8_model)
describe(float_interp, "float32")
describe(int8_interp, "int8")

# 3. 生成测试集，比较 Keras / float32 TFLite / int8 TFLite。
test_rng = np.random.default_rng(456)
x_test, y_test = make_images(200, test_rng)

# Keras 原始模型：输入 float32，输出 0~1 概率。
keras_probs = model.predict(x_test, verbose=0).ravel()

# float32 TFLite：输入 float32，输出 float32。
float_input = float_interp.get_input_details()[0]
float_output = float_interp.get_output_details()[0]
float_probs = []

for image in x_test:
    # 加 batch 维：(16,16,1) -> (1,16,16,1)
    float_interp.set_tensor(float_input["index"], image[None, ...])
    float_interp.invoke()
    float_probs.append(float_interp.get_tensor(float_output["index"])[0, 0])

float_probs = np.array(float_probs)

# int8 TFLite：输入 int8，输出 int8。
# 需要先量化输入，再把输出反量化。
int8_input = int8_interp.get_input_details()[0]
int8_output = int8_interp.get_output_details()[0]

input_scale, input_zero = int8_input["quantization"]
output_scale, output_zero = int8_output["quantization"]

# int8 输入的量化公式：
#   q = round(x / scale) + zero_point
# 然后裁剪到 int8 的 [-128, 127]。
def quantize_input(image):
    q = np.round(image / input_scale) + input_zero
    return np.clip(q, -128, 127).astype(np.int8)


# int8 输出的反量化公式：
#   x = (q - zero_point) * scale
def dequantize_output(q):
    return (float(q) - output_zero) * output_scale


int8_probs = []
for image in x_test:
    int8_interp.set_tensor(int8_input["index"], quantize_input(image)[None, ...])
    int8_interp.invoke()
    raw = int8_interp.get_tensor(int8_output["index"])[0, 0]
    int8_probs.append(dequantize_output(raw))

int8_probs = np.array(int8_probs)

# 概率 >= 0.5 判为竖条（1）。
keras_pred = (keras_probs >= 0.5).astype(np.int32)
float_pred = (float_probs >= 0.5).astype(np.int32)
int8_pred = (int8_probs >= 0.5).astype(np.int32)

print("\naccuracy:")
print("  keras:          ", float(np.mean(keras_pred == y_test)))
print("  float32 tflite: ", float(np.mean(float_pred == y_test)))
print("  int8 tflite:    ", float(np.mean(int8_pred == y_test)))

print("\nprobability difference:")
print("  max |keras - float32|:", float(np.max(np.abs(keras_probs - float_probs))))
print("  max |float32 - int8|: ", float(np.max(np.abs(float_probs - int8_probs))))
print("  mean |float32 - int8|:", float(np.mean(np.abs(float_probs - int8_probs))))

print("\nfirst 5 test samples:")
for i in range(5):
    print(
        f"true={int(y_test[i])}  "
        f"keras={keras_probs[i]:.4f}  "
        f"float32={float_probs[i]:.4f}  "
        f"int8={int8_probs[i]:.4f}"
    )

print("\nsaved:")
print(" ", FLOAT_PATH)
print(" ", INT8_PATH)
