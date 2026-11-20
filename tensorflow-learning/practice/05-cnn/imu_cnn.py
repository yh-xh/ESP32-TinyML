"""用 1D CNN 分辨振动信号来自 x、y 还是 z 轴。

输入是 200 个时间点 x 3 个轴，输出是 3 分类。
这个 shape 和 C5 上的 IMU 手势数据一致：
    200 x 3  ->  Conv1D  ->  MaxPooling1D  ->  Conv1D  ->  GAP  ->  Dense(3)

注释里的 N 表示 batch 大小。model.summary() 里显示 None，是因为
Keras 不在搭模型时固定 batch，训练时喂 32 就是 32，预测时喂 1 就是 1。

Run:
    conda activate tf-esp
    python imu_cnn.py
"""

import numpy as np
import tensorflow as tf
import matplotlib

# 不打开窗口，只保存图片；必须在 import pyplot 之前设置。
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 固定 Python / NumPy / TensorFlow 的随机种子。
tf.keras.utils.set_random_seed(42)

# NumPy 的随机数生成器，专门负责生成合成数据。
rng = np.random.default_rng(0)

CLASS_NAMES = ["x-axis", "y-axis", "z-axis"]

# 200 个时间点，范围 0~2π。
# np.linspace 返回 shape (200,)，后面会广播到 (200, 3) 的某一个轴上。
t = np.linspace(0.0, 2.0 * np.pi, 200, dtype=np.float32)


def make_signals(n):
    """生成 n 条 200x3 信号：正弦波出现在哪个轴，就属于哪一类。

    signals 的 shape 是 (n, 200, 3)：
        n     batch 大小
        200   时间步
        3     通道数（x / y / z 三个轴）

    labels 的 shape 是 (n,)，每个元素是 0、1 或 2。
    """
    # 全 0 数组，shape (n, 200, 3)。
    signals = np.zeros((n, 200, 3), dtype=np.float32)

    # rng.integers(0, 3, size=n) 生成 0/1/2，作为三个类别。
    labels = rng.integers(0, 3, size=n).astype(np.int32)

    for i in range(n):
        # 选一个轴：0=x，1=y，2=z。
        axis = labels[i]

        # 随机相位：0~2π 的标量，让每条波形起点不同。
        phase = rng.uniform(0.0, 2.0 * np.pi)

        # 随机幅度：0.8~1.2 的标量，避免所有样本幅度完全一样。
        amplitude = rng.uniform(0.8, 1.2)

        # t 的 shape 是 (200,)，sin(...) 也是 (200,)。
        # signals[i, :, axis] 的 shape 也是 (200,)，
        # 所以可以把整条波形直接赋给某一个轴。
        signals[i, :, axis] = amplitude * np.sin(2.0 * t + phase)

    # 加噪声：
    # rng.normal(均值, 标准差, shape) 返回和 signals 同 shape 的随机噪声。
    # 三个轴都加，避免模型只记住完全干净的波形。
    signals += rng.normal(0.0, 0.1, signals.shape).astype(np.float32)
    return signals, labels


# 1. 造数据并切分
# 600 条信号，按 480 / 60 / 60 切成训练、验证、测试。
x_all, y_all = make_signals(600)

# permutation 返回随机排列的下标，用同一个 idx 同时打乱 x 和 y。
idx = rng.permutation(len(x_all))
x_all, y_all = x_all[idx], y_all[idx]

x_train, y_train = x_all[:480], y_all[:480]
x_val, y_val = x_all[480:540], y_all[480:540]
x_test, y_test = x_all[540:], y_all[540:]

print("train signals:", x_train.shape, "labels:", y_train.shape)
print("val signals:  ", x_val.shape)
print("test signals: ", x_test.shape)

# 2. 搭 1D CNN
model = tf.keras.Sequential([
    # Input 只声明单个样本的 shape，不包含 batch 维。
    # 每个样本是 200 个时间步、3 个通道。
    tf.keras.layers.Input((200, 3)),

    # Conv1D 参数：
    #   16        filters，输出 16 个通道，也就是 16 个卷积核
    #   9         kernel_size，每个卷积核看连续 9 个时间点
    #   activation="relu"  每个卷积结果过 ReLU
    #   padding="same"     stride=1 时输出时间长度和输入一样
    #
    # 输入 (N,200,3) -> 输出 (N,200,16)
    # 参数：9*3*16 + 16 = 448
    tf.keras.layers.Conv1D(16, 9, activation="relu", padding="same"),

    # MaxPooling1D(2)：每 2 个时间点取最大值，时间长度减半。
    # 默认 strides 也等于 2，所以 200 -> 100。
    #
    # 输入 (N,200,16) -> 输出 (N,100,16)
    tf.keras.layers.MaxPooling1D(2),

    # 第二层卷积：
    #   filters=32
    #   kernel_size=5
    #   输入通道是上一层的 16，所以卷积核实际 shape 是 (5, 16, 32)
    #
    # 输入 (N,100,16) -> 输出 (N,100,32)
    # 参数：5*16*32 + 32 = 2592
    tf.keras.layers.Conv1D(32, 5, activation="relu", padding="same"),

    # GlobalAveragePooling1D：
    # 对时间维求平均，把 (N,100,32) 变成 (N,32)。
    # 每个通道得到一个数，表示这个通道在整段时间里的平均强度。
    # 没有可训练参数。
    tf.keras.layers.GlobalAveragePooling1D(),

    # Dense 参数：
    #   units=3，输出 3 个数
    #   activation="softmax" 把 3 个数变成概率，并且和为 1
    # 输入 (N,32) -> 输出 (N,3)
    # 参数：32*3 + 3 = 99
    tf.keras.layers.Dense(3, activation="softmax"),
])

# compile 只配置训练规则，不训练。
model.compile(
    # Adam 优化器，学习率 1e-3。
    optimizer=tf.keras.optimizers.Adam(1e-3),

    # 标签是整数 0/1/2，所以用 sparse_categorical_crossentropy。
    # 如果标签是 one-hot（[1,0,0] 这种），才用 categorical_crossentropy。
    loss="sparse_categorical_crossentropy",

    # accuracy 只用于监控，不参与梯度更新。
    metrics=["accuracy"],
)

# 打印每层输出 shape 和参数量。
model.summary()

# 3. 训练
history = model.fit(
    # 训练数据：x 是 (480,200,3)，y 是 (480,)。
    x_train,
    y_train,

    # 验证数据：每个 epoch 结束后前向一遍，算 val_loss。
    # 验证不更新参数，只用于监控和 EarlyStopping。
    validation_data=(x_val, y_val),

    # 最多训练 30 个 epoch。
    epochs=30,

    # 每次拿 32 条信号更新一次参数。
    # 480 / 32 = 15 个 batch。
    batch_size=32,

    # 0 不打印；1 进度条；2 每个 epoch 一行。
    verbose=2,

    callbacks=[
        # monitor：监控 val_loss
        # patience：连续 5 轮没改善就停
        # restore_best_weights：停止后恢复到验证 loss 最好的那轮
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ],
)

# 4. 测试集评估
# evaluate 返回 [loss, accuracy]；verbose=0 不打印进度。
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\ntest loss:", float(test_loss))
print("test accuracy:", float(test_acc))

# predict 返回 shape (60, 3) 的概率矩阵。
# 每一行是“这条信号属于 x/y/z”的三个概率，和为 1。
probs = model.predict(x_test, verbose=0)

# argmax(axis=1)：在每行的 3 个概率里取最大值的下标。
# 结果 shape 是 (60,)，元素是 0/1/2。
pred = np.argmax(probs, axis=1)

# 混淆矩阵：行是真实类别，列是预测类别。
# cm[1,2] 表示真实 y、预测 z 的数量。
cm = np.zeros((3, 3), dtype=np.int32)
for true_label, pred_label in zip(y_test, pred):
    cm[true_label, pred_label] += 1

print("\nconfusion matrix (rows=true, cols=pred):")
print(cm)

print("\nfirst 5 test samples:")
for i in range(5):
    # np.round(..., 4) 把三个概率保留 4 位小数，方便打印。
    print(
        f"true={CLASS_NAMES[y_test[i]]:8s} "
        f"pred={CLASS_NAMES[pred[i]]:8s} "
        f"probs={np.round(probs[i], 4)}"
    )

# 5. 保存模型
model.save("imu_cnn.keras")
print("\nsaved: imu_cnn.keras")

# 6. 每个类别画一条测试信号
# 3 行 1 列，sharex=True 表示三张图共用 x 轴刻度。
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

for class_id, ax in enumerate(axes):
    # np.where(y_test == class_id)[0] 找出属于当前类别的所有下标。
    # [0] 取第一个，再取 [0] 得到一个整数下标。
    sample_index = int(np.where(y_test == class_id)[0][0])

    # x_test[sample_index] 的 shape 是 (200,3)。
    # 分别画出 x、y、z 三个通道。
    ax.plot(x_test[sample_index, :, 0], label="x")
    ax.plot(x_test[sample_index, :, 1], label="y")
    ax.plot(x_test[sample_index, :, 2], label="z")

    ax.set_title(f"true class: {CLASS_NAMES[class_id]}")
    ax.set_ylabel("value")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("time step")
fig.tight_layout()
fig.savefig("imu_samples.png", dpi=150)
plt.close(fig)

# 画 loss 和 accuracy 曲线
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# history.history 是字典，键包括：
#   loss / accuracy / val_loss / val_accuracy
# 每个值都是长度为 epoch 数的列表。
axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="validation")
axes[0].set_title("loss")
axes[0].set_xlabel("epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history["accuracy"], label="train")
axes[1].plot(history.history["val_accuracy"], label="validation")
axes[1].set_title("accuracy")
axes[1].set_xlabel("epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("imu_training.png", dpi=150)
plt.close(fig)
print("saved: imu_samples.png, imu_training.png")
