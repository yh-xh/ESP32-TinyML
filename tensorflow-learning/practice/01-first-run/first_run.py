"""第一次跑 TensorFlow：看看张量长什么样。

这一节不训练模型，只做三件事：
1. 导入 TensorFlow；
2. 创建标量、向量、矩阵；
3. 看它们的 shape、dtype、ndim。

Run:
    conda activate tf-esp
    python first_run.py
"""

# import 把 TensorFlow 库加载进来。
# as tf 是起别名，后面写 tf.constant 比写 tensorflow.constant 短。
# 这一步只从本地环境加载库，不需要联网；第一次会慢几秒。
import tensorflow as tf

# __version__ 是 TensorFlow 模块的一个属性，用来确认当前环境版本。
print("TensorFlow:", tf.__version__)

# tf.constant 创建“常量张量”：创建后值不能再改。
# 你可以先把它理解成 TensorFlow 版的数组。
#
# dtype 会根据 Python 字面量自动推断：
#   7              -> int32（整数）
#   [1.0, 2.0]     -> float32（带小数点）
#   [[1, 2], ...]  -> int32（都是整数）
scalar = tf.constant(7)
vector = tf.constant([1.0, 2.0, 3.0])
matrix = tf.constant([[1, 2], [3, 4]])

# 这是一个 Python 循环：
# 列表里放了三个 (名字, 张量) 元组，
# 每轮把元组拆成 name 和 value 两个变量。
#
# 元组就是用小括号包起来的一组值，例如 ("scalar", scalar)。
for name, value in [("scalar", scalar), ("vector", vector), ("matrix", matrix)]:
    # f-string：把变量塞进字符串里。
    # \n 表示换行。
    print(f"\n{name}:")

    # 打印张量本身。TensorFlow 会显示值、shape 和 dtype。
    print(value)

    # shape 是“每一维多大”：
    #   标量 ()        0 维
    #   向量 (3,)      1 维，长度 3
    #   矩阵 (2, 2)    2 维，2 行 2 列
    print("shape:", value.shape)

    # dtype 是元素类型，例如 int32、float32。
    # 模型对 dtype 很敏感，后面经常要检查。
    print("dtype:", value.dtype)

    # ndim 是维度的个数：
    #   标量 0，向量 1，矩阵 2。
    print("ndim :", value.ndim)
