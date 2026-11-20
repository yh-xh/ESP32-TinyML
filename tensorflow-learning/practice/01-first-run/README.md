# 先跑起来：看看 TensorFlow 长什么样

这一节不搭模型，只做一件事：让 Python 和 TensorFlow 跑起来，看看张量到底长什么样。

代码里每一行都写了注释；这份文档只讲跑通之后该看什么、容易卡在哪里。

## 跑一下

```bash
conda activate tf-esp
cd practice/01-first-run
python first_run.py
```

## 跑出来大概是这样

前面可能有一些 CPU / oneDNN 日志，正常。关键部分：

```text
TensorFlow: 2.15.1

scalar:
tf.Tensor(7, shape=(), dtype=int32)
shape: ()
dtype: <dtype: 'int32'>
ndim : 0

vector:
tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32)
shape: (3,)
dtype: <dtype: 'float32'>
ndim : 1

matrix:
tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32)
shape: (2, 2)
dtype: <dtype: 'int32'>
ndim : 2
```

## 只记三件事

| 属性 | 含义 | 例子 |
|---|---|---|
| `shape` | 每一维多大 | `(2, 2)` |
| `dtype` | 元素类型 | `int32` / `float32` |
| `ndim` | 有几个维度 | 标量 0，向量 1，矩阵 2 |

最容易错的是 shape：

```text
()        标量，一个数
(3,)      长度为 3 的向量
(3, 1)    3 行 1 列
(1, 3)    1 行 3 列
```

`()` 不是 `(1,)`。后面模型报 shape 错误时，第一反应就是打印 shape。

## 常见报错

如果看到：

```text
ModuleNotFoundError: No module named 'tensorflow'
```

先检查 Python 是不是在 `tf-esp` 环境里：

```bash
conda activate tf-esp
which python
```

`which python` 应该指向 `tf-esp` 环境。

## 顺手改两下

1. 把 `scalar` 改成 `3.14`，看 `dtype` 变成什么；
2. 新建一个 `(3, 2)` 的矩阵，打印 shape；
3. 用 `tf.reshape(vector, (1, 3))` 把向量变成 `(1, 3)`；
4. 打印 `tf.rank(matrix)`，和 `matrix.ndim` 对比。

## 到这儿你应该能

- 自己把脚本跑起来；
- 解释 `shape`、`dtype`、`ndim`；
- 说出 `()`、`(3,)`、`(2, 2)` 分别代表什么；
- 遇到 `ModuleNotFoundError` 知道先检查环境。

下一步是索引、切片、reshape、矩阵乘和广播。
