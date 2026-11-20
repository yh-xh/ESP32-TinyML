"""用一个小 CNN 分辨“横条”和“竖条”。

输入是 16x16 的灰度图，输出是二分类。
这一节只关心：
1. Conv2D / MaxPooling2D / Flatten 的 shape 怎么变；
2. 卷积的参数怎么算；
3. 二分类的最后一层和 loss 怎么配。

注释里的 N 表示 batch 大小。model.summary() 里显示 None，是因为
Keras 不在搭模型时固定 batch，训练时喂 32 就是 32，预测时喂 1 就是 1。

Run:
    conda activate tf-esp
    python image_cnn.py
"""

import numpy as np
import tensorflow as tf
import matplotlib

# Agg 表示不打开窗口，只把图保存成文件。
# 必须在 import pyplot 之前设置，否则有些环境会去找显示器。
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 固定 Python / NumPy / TensorFlow 的随机种子。
# 这样权重初始化、数据打乱、Dropout 等随机过程每次尽量一致。
tf.keras.utils.set_random_seed(42)

# NumPy 自己的随机数生成器。
# 后面的 randint / permutation / uniform 都从它来，便于单独控制数据随机性。
rng = np.random.default_rng(0)

LABEL_NAMES = ["horizontal", "vertical"]


def make_images(n):
    """生成 n 张 16x16 灰度图：横条是 0，竖条是 1。

    images 的 shape 是 (n, 16, 16, 1)：
        n     batch 大小
        16    高
        16    宽
        1     通道数（灰度图是 1，RGB 图是 3）

    labels 的 shape 是 (n,)，每个元素是 0 或 1。
    """
    # np.zeros 创建全 0 数组；dtype=np.float32 和 TensorFlow 默认类型一致。
    images = np.zeros((n, 16, 16, 1), dtype=np.float32)

    # rng.integers(0, 2, size=n) 生成 [0, 2) 的整数，也就是 0 或 1。
    # astype(np.int32) 把类型固定成整数，后面做混淆矩阵时方便。
    labels = rng.integers(0, 2, size=n).astype(np.int32)

    for i in range(n):
        # 条形的厚度：2 或 3 个像素。
        # rng.integers(2, 4) 的上界 4 不包含，所以结果只有 2、3。
        thickness = int(rng.integers(2, 4))

        if labels[i] == 0:
            # 横条：随机选一个起始行，把这几行的所有列涂成 1。
            # 上界 16 - thickness + 1 保证 row + thickness 不超过 16。
            row = int(rng.integers(0, 16 - thickness + 1))

            # 切片含义：
            # images[i, 行范围, 所有列, 第 0 个通道] = 1.0
            images[i, row:row + thickness, :, 0] = 1.0
        else:
            # 竖条：随机选一个起始列，把这几列的所有行涂成 1。
            col = int(rng.integers(0, 16 - thickness + 1))
            images[i, :, col:col + thickness, 0] = 1.0

    return images, labels


# 1. 造数据并切分
# 一共 1000 张图。
x_all, y_all = make_images(1000)

# rng.permutation(1000) 返回 0~999 的一个随机排列。
# 用同一个 idx 同时打乱图片和标签，保证 x 和 y 仍然一一对应。
idx = rng.permutation(len(x_all))
x_all, y_all = x_all[idx], y_all[idx]

# 前 800 张训练，中间 100 张验证，最后 100 张测试。
# 验证集参与 EarlyStopping 的决策；测试集只在最后评估一次。
x_train, y_train = x_all[:800], y_all[:800]
x_val, y_val = x_all[800:900], y_all[800:900]
x_test, y_test = x_all[900:], y_all[900:]

print("train images:", x_train.shape, "labels:", y_train.shape)
print("val images:  ", x_val.shape)
print("test images: ", x_test.shape)

# 2. 搭 CNN
# Sequential 表示“按列表顺序一层一层往下传”。
model = tf.keras.Sequential([
    # Input 只声明“单个样本”的 shape，不包含 batch 维。
    # 所以这里写 (16, 16, 1)，不是 (N, 16, 16, 1)。
    # 训练时 Keras 自动补上 batch：实际输入是 (32, 16, 16, 1)。
    tf.keras.layers.Input((16, 16, 1)),

    # Conv2D 参数：
    #   8          filters，输出通道数，也就是 8 个不同的卷积核
    #   (3, 3)     kernel_size，每个卷积核看 3x3 的局部区域
    #   activation 每个卷积结果先过 ReLU：max(0, z)
    #   padding="same"  在边缘补零，stride=1 时输出高宽和输入一样
    #
    # 输入 (N,16,16,1) -> 输出 (N,16,16,8)
    # 参数：3*3*1*8 + 8 = 80
    tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),

    # MaxPooling2D 参数：
    #   pool_size=(2,2)  每个 2x2 小块取最大值
    #   strides 默认等于 pool_size，所以高宽各减半
    # 没有可训练参数。
    #
    # 输入 (N,16,16,8) -> 输出 (N,8,8,8)
    tf.keras.layers.MaxPooling2D((2, 2)),

    # 第二层卷积：
    #   filters=16，输出 16 个通道
    #   kernel_size=(3,3)
    #   输入通道是上一层的 8，所以实际的卷积核 shape 是 (3,3,8,16)
    #
    # 输入 (N,8,8,8) -> 输出 (N,8,8,16)
    # 参数：3*3*8*16 + 16 = 1168
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),

    # 输入 (N,8,8,16) -> 输出 (N,4,4,16)
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten 把 (N,4,4,16) 拉平成 (N,256)。
    # 4*4*16 = 256，数据顺序按行优先展开。
    tf.keras.layers.Flatten(),

    # Dense 参数：
    #   units=1，输出 1 个数
    #   activation="sigmoid" 把输出压到 0~1，当作“是竖条”的概率
    # 输入 (N,256) -> 输出 (N,1)
    # 参数：256*1 + 1 = 257
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

# compile 只配置训练规则，不训练。
model.compile(
    # Adam 是优化器；1e-3 是学习率，也就是每次参数更新的步长系数。
    optimizer=tf.keras.optimizers.Adam(1e-3),

    # binary_crossentropy 是二分类交叉熵。
    # 标签是 0/1，模型输出是 0~1 的概率，正好配套。
    loss="binary_crossentropy",

    # accuracy 只用于打印和监控，不参与梯度更新。
    metrics=["accuracy"],
)

# 打印每层输出 shape 和参数量。
# summary() 自己会打印，返回值是 None，不要再 print(model.summary())。
model.summary()

# 3. 训练
# fit 返回 History 对象，里面保存了每个 epoch 的 loss / accuracy。
history = model.fit(
    # 训练数据：第一个维度都是样本数。
    x_train,
    y_train,

    # 验证数据：每个 epoch 训练结束后，用这块数据前向一遍，算 val_loss。
    # 验证不更新参数，只用于监控和 EarlyStopping。
    validation_data=(x_val, y_val),

    # 最多训练 20 个 epoch（把训练集完整过 20 遍）。
    epochs=20,

    # 每次拿 32 个样本算 loss、更新一次参数。
    # 800 张训练图 -> 800/32 = 25 个 batch。
    batch_size=32,

    # 日志级别：
    #   0 不打印
    #   1 进度条
    #   2 每个 epoch 一行
    verbose=2,

    # 回调：在训练过程中按条件自动执行的动作。
    callbacks=[
        # EarlyStopping 参数：
        #   monitor="val_loss"        盯验证 loss
        #   patience=5                连续 5 轮没改善就停
        #   restore_best_weights=True 停止后恢复到验证 loss 最好的那轮
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ],
)

# 4. 测试集评估
# evaluate 返回 [loss, accuracy]；verbose=0 表示不打印进度。
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\ntest loss:", float(test_loss))
print("test accuracy:", float(test_acc))

# predict 返回 shape (100, 1) 的概率数组。
# ravel() 把 (100,1) 拉成 (100,)，方便逐个处理。
prob = model.predict(x_test, verbose=0).ravel()

# 概率 >= 0.5 判为竖条（1），否则判为横条（0）。
pred = (prob >= 0.5).astype(np.int32)

# 标签本来就是 int32，这里再转一次只是让类型明确。
truth = y_test.astype(np.int32)

# 混淆矩阵：行是真实类别，列是预测类别。
# cm[0,0] 表示真实横条、预测横条的数量。
cm = np.zeros((2, 2), dtype=np.int32)
for t, p in zip(truth, pred):
    cm[t, p] += 1

print("\nconfusion matrix (rows=true, cols=pred):")
print(cm)

print("\nfirst 5 test samples:")
for i in range(5):
    # :10s 表示字符串左对齐、至少占 10 个字符。
    print(f"true={LABEL_NAMES[truth[i]]:10s}  p(vertical)={prob[i]:.4f}")

# 5. 保存模型
# .keras 文件里包含结构、权重和 compile 配置。
model.save("image_cnn.keras")
print("\nsaved: image_cnn.keras")

# 6. 画几张测试图
# subplots(2, 4) 返回：
#   fig  整个画布
#   axes 2x4 的坐标轴数组
# ravel() 把 axes 展平成一维，方便 zip。
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for ax, image, label in zip(axes.ravel(), x_test[:8], truth[:8]):
    # image 的 shape 是 (16,16,1)，imshow 需要二维，所以取 [...,0]。
    # cmap="gray_r" 是反向灰度：0 显示白色，1 显示黑色。
    # vmin/vmax 固定颜色映射范围，避免自动拉伸。
    ax.imshow(image[..., 0], cmap="gray_r", vmin=0.0, vmax=1.0)
    ax.set_title(LABEL_NAMES[label])
    ax.axis("off")  # 不显示坐标轴刻度
fig.suptitle("sample test images")
fig.tight_layout()  # 自动调整子图间距
fig.savefig("image_samples.png", dpi=150)
plt.close(fig)  # 释放画布，避免内存越积越多

# 画 loss 和 accuracy 曲线
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# history.history 是一个字典，键包括：
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
fig.savefig("image_training.png", dpi=150)
plt.close(fig)
print("saved: image_samples.png, image_training.png")
