"""检查一个 TFLite 模型的输入、输出、量化参数和算子。

用在 08 这种“换模型”场景：
先确认新模型和板端工程期望的输入/输出、算子对得上，再改固件。

Run:
    conda activate tf-esp
    cd practice/08-camera
    python check_model.py ../06-tflite/image_cnn_int8.tflite
"""

import argparse
from pathlib import Path

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help=".tflite 文件路径")
    args = parser.parse_args()

    data = args.model.read_bytes()
    print("model:", args.model)
    print("size: ", len(data), "bytes")

    # Interpreter 只解析模型，不训练；allocate_tensors 会根据模型分配张量。
    interpreter = tf.lite.Interpreter(model_content=data)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    print("\ninput:")
    print("  shape:       ", list(input_detail["shape"]))
    print("  dtype:       ", input_detail["dtype"])
    print("  quantization:", input_detail["quantization"])

    print("\noutput:")
    print("  shape:       ", list(output_detail["shape"]))
    print("  dtype:       ", output_detail["dtype"])
    print("  quantization:", output_detail["quantization"])

    # _get_ops_details() 是 TFLite Python 的解释器接口，用来列出实际算子。
    # PC 上可能多一个 DELEGATE（XNNPACK），它不是模型本身的算子。
    ops = sorted({
        op["op_name"]
        for op in interpreter._get_ops_details()
        if op["op_name"] != "DELEGATE"
    })
    print("\nops:")
    for op in ops:
        print(" ", op)

    # tflite_person 的参考工程使用 96x96 灰度、2 类输出。
    # 这个检查不等于模型一定能用，只是提前发现最常见的 shape 不一致。
    expected_input = [1, 96, 96, 1]
    if list(input_detail["shape"]) != expected_input:
        print("\n注意: tflite_person 期望输入", expected_input,
              "，当前是", list(input_detail["shape"]))
    if len(output_detail["shape"]) != 2:
        print("注意: 分类模型通常输出 (1, 类别数)，当前输出 shape 是",
              list(output_detail["shape"]))


if __name__ == "__main__":
    main()
