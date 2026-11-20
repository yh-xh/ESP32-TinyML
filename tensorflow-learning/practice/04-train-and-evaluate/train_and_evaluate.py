"""验证集、过拟合与 EarlyStopping。

训练数据故意只给 6 个点，让过拟合在几百个 epoch 内出现：
训练 loss 会降到接近 0，但验证 loss 在某个 epoch 之后开始变差。

注释里写的是每个 API 的参数含义；
文档里讲为什么这样分数据、怎么读曲线、什么时候该早停。

Run:
    conda activate tf-esp
    python train_and_evaluate.py
"""

import numpy as np
import tensorflow as tf
import matplotlib

# Agg 表示不打开窗口，只保存图片。
# 必须在 import pyplot 之前设置。
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 固定随机种子，让每次运行的初始化、数据打乱尽量一致。
tf.keras.utils.set_random_seed(42)

# NumPy 随机数生成器，用来生成测试集的随机点。
rng = np.random.default_rng(0)

# 1. 训练集：只有 6 个点
# linspace 均匀取 6 个点，reshape 成 (6,1)。
# 数据太少，1,153 个参数的模型足以把这 6 个点背下来。
x_train = np.linspace(0.0, 2.0 * np.pi, 6, dtype=np.float32).reshape(-1, 1)
y_train = np.cos(x_train).astype(np.float32)

# 2. 验证集：200 个点，密集覆盖 [0, 2π]
# 每个 epoch 结束后，模型会在这块数据上前向一遍，算 val_loss。
# 验证集参与 EarlyStopping / ModelCheckpoint 的决策，但不更新参数。
x_val = np.linspace(0.0, 2.0 * np.pi, 200, dtype=np.float32).reshape(-1, 1)
y_val = np.cos(x_val).astype(np.float32)

# 3. 测试集：200 个随机点
# rng.uniform(起点, 终点, 个数) 生成随机数。
# 训练和调参时都不看它，最后只 evaluate 一次。
x_test = rng.uniform(0.0, 2.0 * np.pi, 200).astype(np.float32).reshape(-1, 1)
y_test = np.cos(x_test).astype(np.float32)

# 4. 模型：结构和 03 一样，1153 个参数。
model = tf.keras.Sequential([
    tf.keras.layers.Input((1,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),
])

model.compile(
    # 学习率 5e-3，比 03 的 1e-3 大，让 6 个点在几百轮内被拟合。
    optimizer=tf.keras.optimizers.Adam(5e-3),
    # 回归任务，用均方误差。
    loss="mse",
)

# 5. 两个回调
# 回调 = 训练过程中按条件自动执行的动作。
early_stopping = tf.keras.callbacks.EarlyStopping(
    # 监控验证 loss，不是训练 loss。
    monitor="val_loss",

    # 连续 50 个 epoch 没有改善就停。
    patience=50,

    # 停止后把权重恢复到 val_loss 最好的那一轮。
    restore_best_weights=True,

    # 1 表示早停时打印一条消息。
    verbose=1,
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    # 保存路径。文件格式由 .keras 后缀决定。
    "best_model.keras",

    # 同样盯 val_loss。
    monitor="val_loss",

    # 只在 val_loss 创新低时保存，避免覆盖成最后一轮。
    save_best_only=True,

    # 1 表示每次保存时打印消息。
    verbose=1,
)

history = model.fit(
    # 训练数据。
    x_train,
    y_train,

    # 每个 epoch 结束后用验证集算 val_loss。
    # 注意验证集只做前向，不参与反向传播。
    validation_data=(x_val, y_val),

    # 最多 500 轮；EarlyStopping 会提前停。
    epochs=500,

    # 6 个训练点正好一个 batch，每个 epoch 更新一次参数。
    batch_size=6,

    # 每个 epoch 打印一行。
    verbose=2,

    # 把两个回调挂上去。
    callbacks=[early_stopping, checkpoint],
)

# history.history["loss"] 和 ["val_loss"] 都是长度等于实际 epoch 数的列表。
# np.array 转成数组，方便 argmin。
train_loss = np.array(history.history["loss"])
val_loss = np.array(history.history["val_loss"])

# argmin 返回最小值的下标；best_epoch 从 0 开始，所以打印时 +1。
best_epoch = int(np.argmin(val_loss))

print("\n--- summary ---")
print("epochs run:", len(train_loss))
print("best epoch:", best_epoch + 1)
print("best val MSE:", float(val_loss[best_epoch]))
print("last epoch train MSE:", float(train_loss[-1]))
print("last epoch val MSE:", float(val_loss[-1]))

# evaluate 返回一个标量 loss（这里没有 metrics）。
# 此时模型已经被 restore_best_weights 恢复成第 155 轮的权重，
# 所以 train_mse 不是最后一轮的 7e-6，而是最好权重下的 0.0025。
train_mse = model.evaluate(x_train, y_train, verbose=0)
val_mse = model.evaluate(x_val, y_val, verbose=0)
test_mse = model.evaluate(x_test, y_test, verbose=0)

# 基线：永远预测训练集平均值，看模型有没有比“瞎猜均值”更好。
# y_train.mean() 是标量，广播到 y_test 的 shape。
baseline_mse = float(np.mean((y_test - y_train.mean()) ** 2))

print("\nMSE after restoring best weights:")
print("train:", float(train_mse))
print("val:  ", float(val_mse))
print("test: ", float(test_mse))
print("baseline test MSE (always predict train mean):", baseline_mse)

# 6. 画两张图
# subplots(1,2) 返回 fig（画布）和 axes（两个坐标轴）。
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左边：训练 / 验证 loss 曲线。
axes[0].plot(train_loss, label="train")
axes[0].plot(val_loss, label="validation")

# 在最好的 epoch 处画一条竖虚线。
# best_epoch 是 0-based，所以横坐标直接用 best_epoch。
axes[0].axvline(best_epoch, color="gray", linestyle="--",
                label=f"best epoch {best_epoch + 1}")

# 对数纵轴：训练 loss 会到 1e-6，验证 loss 在 1e-2，
# 线性刻度下训练曲线会被压扁。
axes[0].set_yscale("log")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("MSE (log scale)")
axes[0].set_title("train vs validation loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 右边：在 1000 个密集点上比较真实 cos 和模型预测。
x_dense = np.linspace(0.0, 2.0 * np.pi, 1000, dtype=np.float32).reshape(-1, 1)
y_dense = np.cos(x_dense)
y_pred = model.predict(x_dense, verbose=0)

axes[1].plot(x_dense, y_dense, label="cos(x)")
axes[1].plot(x_dense, y_pred, label="prediction")

# 红点标出 6 个训练点，看模型有没有穿过它们。
axes[1].scatter(x_train, y_train, color="tab:red", zorder=3, label="train points")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("prediction on dense grid")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("training_curve.png", dpi=150)
plt.close(fig)
print("\nsaved plot: training_curve.png")
print("saved best model: best_model.keras")
