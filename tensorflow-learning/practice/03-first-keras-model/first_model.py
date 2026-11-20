"""第一次用 Keras 训练一个模型。

目标函数：y = cos(x)，x 在 [0, 2*pi]。
这一节只跑通最基本的一条链路：
    造数据 -> 搭模型 -> compile -> fit -> predict -> save

注释里写的是每个 API 在干什么、参数是什么意思；
验证集、测试集、EarlyStopping 留到下一节，不在这里堆。

Run:
    conda activate tf-esp
    python first_model.py
"""

import numpy as np
import tensorflow as tf

# np.linspace(起点, 终点, 点数, dtype=类型)
# 在 [0, 2π] 里均匀取 10000 个点，返回 shape (10000,)。
# 2.0 * np.pi 就是 2π，约等于 6.283。
x = np.linspace(0.0, 2.0 * np.pi, 10000, dtype=np.float32).reshape(-1, 1)

# np.cos 逐元素计算余弦，输出 shape 仍然是 (10000,1)。
# astype(np.float32) 显式固定类型，和 TensorFlow 默认的 float32 对齐。
# y 的范围是 [-1, 1]。
y = np.cos(x).astype(np.float32)

# Sequential：按列表顺序一层一层传数据。
model = tf.keras.Sequential([
    # Input 只声明“单个样本”的 shape，不包含 batch 维。
    # (1,) 表示每个样本是 1 个数；实际训练时 Keras 自动加 batch，
    # 例如 64 个样本就是 (64,1)。
    tf.keras.layers.Input((1,)),

    # Dense 是全连接层：
    #   units=32          这一层有 32 个神经元
    #   activation=relu   每个神经元先算 x@W+b，再过 max(0,z)
    #
    # 输入 (N,1) -> 输出 (N,32)
    # 参数：1*32 个权重 + 32 个偏置 = 64
    tf.keras.layers.Dense(32, activation="relu"),

    # 第二层：
    # 输入 (N,32) -> 输出 (N,32)
    # 参数：32*32 + 32 = 1056
    tf.keras.layers.Dense(32, activation="relu"),

    # 最后一层：
    #   units=1，输出 1 个数
    #   不写 activation，表示线性输出，可以是任意实数
    #   cos 有负值，所以不能用 ReLU（会把负数截成 0）
    #
    # 输入 (N,32) -> 输出 (N,1)
    # 参数：32*1 + 1 = 33
    tf.keras.layers.Dense(1),
])

# compile 只配置训练规则，不训练。
model.compile(
    # optimizer：怎么更新参数。
    # Adam 是常用优化器；1e-3 是学习率，控制每次更新的步长。
    optimizer=tf.keras.optimizers.Adam(1e-3),

    # loss：怎么衡量预测和目标的差距。
    # mse 是均方误差 mean((pred - y)^2)，回归任务常用。
    loss="mse",
)

# fit 真正开始训练，返回 History 对象。
history = model.fit(
    # 训练输入和标签，第一个维度都是样本数 10000。
    x,
    y,

    # 把全部数据重复 200 遍。
    epochs=200,

    # 每次拿 64 个样本算一次 loss、更新一次参数。
    # 10000/64 -> 每个 epoch 157 个 batch，最后一次只有 16 个样本。
    batch_size=64,

    # 0 不打印；1 进度条；2 每个 epoch 一行。
    # 之前用 0 时看起来像卡住，就是因为训练过程完全静默。
    verbose=2,
)

# history.history 是字典，键包括 "loss"（这里没有验证集，所以没有 val_loss）。
# [-1] 取最后一个 epoch 的 loss。
print("final training loss:", history.history["loss"][-1])

# 打印每层输出 shape 和参数量。
# summary() 自己会打印，返回值是 None。
print("model summary:")
model.summary()

print("\npredictions:")
for value in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]:
    # predict 的输入必须带 batch 维。
    # np.array([[value]]) 的 shape 是 (1,1)：1 个样本、1 个特征。
    # 返回 shape 也是 (1,1)，[0,0] 取出那个标量。
    pred = model.predict(np.array([[value]], dtype=np.float32), verbose=0)[0, 0]

    # 理论值，用来和预测对比。
    expected = np.cos(value)
    print(f"x={value:.3f}  pred={pred:.3f}  expected={expected:.3f}")

# 保存成 .keras 文件。
# 文件里包含模型结构、权重和 compile 配置，后面可以 load_model 重新加载。
model.save("cos_model.keras")
print("\nsaved: cos_model.keras")
